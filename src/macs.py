"""
macs.py  -  count Multiply-Accumulate operations (MACs) of DistilBERT.

    python -m src.macs

How:
  * PyTorch's built-in FlopCounterMode counts the FLOPs of every matrix
    multiplication in one forward pass. 1 MAC = 2 FLOPs, so MACs = FLOPs / 2.
    (Element-wise ops like softmax/GELU/LayerNorm are not counted - this is the
    usual convention for MACs.)
  * Forward hooks on every nn.Linear record how many MACs each Linear layer
    does. This tells us which share of all MACs runs inside the layers that
    dynamic quantization converts.
  * The result is double-checked with a hand-derived formula (analytic_macs).

MACs depend only on the sequence length L, so for the dev-set average we run
one forward pass per distinct length and weight it by how often it occurs.

Dynamic quantization does NOT change the number of MACs - it changes the
precision in which the Linear-layer MACs are executed. So this is computed
once, on the FP32 model, and reused for all methods.
"""
import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from transformers.utils import logging as hf_logging

from src import config
from src.data import load_sst2_dev, load_tokenizer, make_single_inputs
from src.quantization import load_fp32_model

hf_logging.set_verbosity_error()


def macs_file(results_dir):
    return os.path.join(results_dir, "macs.json")


def analytic_macs(model_config, seq_len):
    """
    Hand-derived MACs for one DistilBERT sequence of length L:
      per layer: Q,K,V,O projections  4 * L * d * d
                 feed-forward         2 * L * d * d_ff
                 Q.K^T and A.V        2 * L * L * d
      head:      pre_classifier + classifier on the [CLS] token only: d*d + d*C
    """
    L = seq_len
    d = getattr(model_config, "dim", None) or model_config.hidden_size
    d_ff = getattr(model_config, "hidden_dim", None) or model_config.intermediate_size
    n_layers = getattr(model_config, "n_layers", None) or model_config.num_hidden_layers
    n_classes = model_config.num_labels
    per_layer = 4 * L * d * d + 2 * L * d * d_ff + 2 * L * L * d
    return n_layers * per_layer + d * d + d * n_classes


@torch.inference_mode()
def profile_one_input(model, inputs):
    """
    Return (total MACs, {linear_layer_name: MACs}) for one forward pass.
    If FlopCounterMode fails (very unusual), total is None and the formula is used.
    """
    linear_macs = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def hook(mod, inp, out, name=name):
                # input has (tokens x in_features) elements -> tokens*in*out MACs
                linear_macs[name] = linear_macs.get(name, 0) + inp[0].numel() * mod.out_features
            hooks.append(module.register_forward_hook(hook))

    try:
        with FlopCounterMode(display=False) as counter:
            model(**inputs)
        total = counter.get_total_flops() // 2
    except Exception as err:  # fallback: hooks still give the Linear MACs
        print(f"[MACs] WARNING: FlopCounterMode failed ({err}); using the analytic formula.")
        linear_macs.clear()
        model(**inputs)
        total = None
    finally:
        for h in hooks:
            h.remove()
    return total, linear_macs


def compute_mac_profile(results_dir=config.RESULTS_DIR, limit=None):
    """Count MACs, save them to results/macs.json and return them."""
    print("[MACs] loading tokenizer, dev set and FP32 model (eager attention)...")
    tokenizer = load_tokenizer()
    sentences, _ = load_sst2_dev(limit)
    single_inputs = make_single_inputs(tokenizer, sentences)
    # "eager" attention = plain matmuls, which FlopCounterMode can always see
    model = load_fp32_model(attn_implementation="eager")

    # (a) one reference input of exactly MAC_REFERENCE_LENGTH tokens
    ref_len = config.MAC_REFERENCE_LENGTH
    enc = tokenizer(sentences[0], padding="max_length", truncation=True,
                    max_length=ref_len, return_tensors="pt")
    ref_inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    ref_total, ref_linear = profile_one_input(model, ref_inputs)
    ref_analytic = analytic_macs(model.config, ref_len)
    if ref_total is None:
        ref_total = ref_analytic

    # (b) average per dev sentence (batch size 1, real lengths)
    length_counts = Counter(x["input_ids"].shape[1] for x in single_inputs)
    example_of_length = {}
    for x in single_inputs:
        example_of_length.setdefault(x["input_ids"].shape[1], x)

    dev_total, dev_analytic, dev_linear = 0, 0, {}
    for seq_len, count in sorted(length_counts.items()):
        total, linear = profile_one_input(model, example_of_length[seq_len])
        formula = analytic_macs(model.config, seq_len)
        dev_total += count * (total if total is not None else formula)
        dev_analytic += count * formula
        for name, macs in linear.items():
            dev_linear[name] = dev_linear.get(name, 0) + count * macs
    n = len(single_inputs)

    profile = {
        "ref_seq_len": ref_len,
        "ref_total_macs": int(ref_total),
        "ref_analytic_macs": int(ref_analytic),
        "ref_linear_macs": {k: int(v) for k, v in ref_linear.items()},
        "n_samples": n,
        "dev_mean_seq_len": sum(L * c for L, c in length_counts.items()) / n,
        "dev_mean_total_macs": dev_total / n,
        "dev_mean_analytic_macs": dev_analytic / n,
        "dev_mean_linear_macs": {k: v / n for k, v in dev_linear.items()},
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(macs_file(results_dir), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    ok = "OK" if ref_total == ref_analytic else "MISMATCH - check!"
    print(f"[MACs] {ref_len} tokens : counter {ref_total/1e9:.3f} G | formula {ref_analytic/1e9:.3f} G  [{ok}]")
    print(f"[MACs] dev average ({profile['dev_mean_seq_len']:.1f} tokens): "
          f"{profile['dev_mean_total_macs']/1e9:.3f} GMACs per sentence")
    print(f"[MACs] saved to {macs_file(results_dir)}")
    return profile


def load_or_compute_mac_profile(results_dir, n_samples, limit=None):
    """Reuse results/macs.json if it was made for the same dev subset, else recompute."""
    path = macs_file(results_dir)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)
        if profile.get("n_samples") == n_samples:
            return profile
    return compute_mac_profile(results_dir, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count DistilBERT MACs")
    parser.add_argument("--results_dir", default=config.RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N dev sentences")
    args = parser.parse_args()
    compute_mac_profile(args.results_dir, args.limit)
