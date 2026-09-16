"""
benchmark.py  -  measure ONE model (FP32 or one quantized variant) on the CPU
and save everything to results/<method>.json

    python -m src.benchmark --method fp32
    python -m src.benchmark --method dq_int8_per_tensor

run_all.py calls this once per method, each time in a FRESH Python process,
so the RAM measurements of different models never mix.

Measurements (task item 2, repeated for every quantized model in item 3):
  a) number of parameters + MACs
  b) latency in ms (batch size 1 and batch size 32)
  c) memory: saved model size, RAM of the loaded model, peak RAM during inference
  d) accuracy (+ macro-F1) on the SST-2 dev set
"""
import argparse
import json
import os
import time

import torch
from transformers.utils import logging as hf_logging

from src import config
from src.data import length_stats, load_sst2_dev, load_tokenizer, make_batches, make_single_inputs
from src.environment import collect_environment
from src.macs import load_or_compute_mac_profile
from src.measure import (PeakRSSMonitor, count_parameters, evaluate_accuracy,
                         latency_batched, latency_single, release_memory, rss_mb,
                         serialized_size_mb)
from src.quantization import (METHOD_DESCRIPTIONS, apply_quantization, check_label_mapping,
                              describe_layers, load_fp32_model, select_quantized_engine)

hf_logging.set_verbosity_error() 


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one (quantized) DistilBERT model")
    parser.add_argument("--method", required=True, choices=list(METHOD_DESCRIPTIONS))
    parser.add_argument("--threads", type=int, default=config.NUM_THREADS)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N dev sentences (quick test)")
    parser.add_argument("--results_dir", default=config.RESULTS_DIR)
    return parser.parse_args()


def step(text):
    print(f"\n--- {text}", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    torch.set_num_threads(args.threads)
    engine = select_quantized_engine()
    os.makedirs(os.path.join(args.results_dir, "preds"), exist_ok=True)

    print("=" * 70)
    print(f"Method : {args.method}  ->  {METHOD_DESCRIPTIONS[args.method]}")
    print(f"Device : CPU | threads = {torch.get_num_threads()} | quantized engine = {engine}")
    print("=" * 70)

    # ---------------------------------------------------------------- data
    step("1) Loading SST-2 dev set and tokenizing (done before any timing)")
    tokenizer = load_tokenizer()
    sentences, labels = load_sst2_dev(args.limit)
    single_inputs = make_single_inputs(tokenizer, sentences)
    batches = make_batches(tokenizer, sentences, labels, config.BATCH_SIZE)
    data_info = length_stats(single_inputs)
    print(f"  {data_info['n_sentences']} sentences, mean {data_info['mean_tokens']:.1f} tokens "
          f"(max {data_info['max_tokens']})")

    # --------------------------------------------------------------- model
    step("2) Loading pretrained FP32 model" + ("" if args.method == "fp32" else " and quantizing"))
    # import the DistilBERT code now, so its import cost is not counted as model RAM
    from transformers import DistilBertForSequenceClassification  
    release_memory()
    rss_before_model = rss_mb()

    model = load_fp32_model()
    check_label_mapping(model)
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    quantization_time_s = 0.0
    if args.method != "fp32":
        start = time.perf_counter()
        quantized = apply_quantization(model, args.method)
        quantization_time_s = time.perf_counter() - start
        del model            
        model = quantized
    release_memory()
    rss_after_model = rss_mb()

    layers = describe_layers(model)
    params = count_parameters(model)
    size_mb = serialized_size_mb(model)
    release_memory()
    print(f"  parameters: {params['total']:,}  stored as {params['by_storage_dtype']}")
    print(f"  saved model size: {size_mb:.1f} MB | RAM of loaded model: "
          f"{rss_after_model - rss_before_model:.1f} MB")

    # ------------------------------------------------------------ accuracy
    step(f"3) Accuracy on the dev set (batch size {config.BATCH_SIZE}) + peak RAM")
    rss_ready = rss_mb()
    with PeakRSSMonitor(config.MEMORY_POLL_INTERVAL_S) as monitor:
        metrics, preds, gold = evaluate_accuracy(model, batches)
    print(f"  accuracy = {100 * metrics['accuracy']:.2f}% "
          f"({metrics['n_correct']}/{metrics['n_samples']}) | macro-F1 = {metrics['macro_f1']:.4f}")
    print(f"  peak RAM during inference: {monitor.peak_mb:.1f} MB (process total)")

    # ------------------------------------------------------------- latency
    step(f"4) Latency, batch size 1 ({len(single_inputs)} sentences, {config.WARMUP_ITERS} warm-up runs)")
    lat_bs1 = latency_single(model, single_inputs, config.WARMUP_ITERS)
    print(f"  mean {lat_bs1['mean_ms']:.2f} +/- {lat_bs1['std_ms']:.2f} ms | "
          f"median {lat_bs1['median_ms']:.2f} ms | p95 {lat_bs1['p95_ms']:.2f} ms")

    step(f"5) Latency, batch size {config.BATCH_SIZE} ({config.BATCHED_REPEATS} passes over the dev set)")
    lat_batched = latency_batched(model, batches, config.BATCHED_WARMUP_ITERS, config.BATCHED_REPEATS)
    print(f"  {lat_batched['ms_per_batch_mean']:.1f} ms per batch | "
          f"{lat_batched['ms_per_sample']:.2f} ms per sentence | "
          f"{lat_batched['throughput_samples_per_s']:.1f} sentences/s")

    # ---------------------------------------------------------------- MACs
    step("6) MACs (counted once on the FP32 graph; quantization does not change the count)")
    mac_profile = load_or_compute_mac_profile(args.results_dir, len(sentences), args.limit)
    quantized_names = set(layers["quantized_linear"])

    def share(linear_macs, total):
        return sum(v for k, v in linear_macs.items() if k in quantized_names) / total

    macs = {
        "per_sentence_dev_mean": mac_profile["dev_mean_total_macs"],
        f"per_sentence_{mac_profile['ref_seq_len']}_tokens": mac_profile["ref_total_macs"],
        "linear_share_dev": sum(mac_profile["dev_mean_linear_macs"].values()) / mac_profile["dev_mean_total_macs"],
        "quantized_share_dev": share(mac_profile["dev_mean_linear_macs"], mac_profile["dev_mean_total_macs"]),
        "quantized_share_ref": share(mac_profile["ref_linear_macs"], mac_profile["ref_total_macs"]),
    }
    print(f"  {macs['per_sentence_dev_mean'] / 1e9:.3f} GMACs per dev sentence; "
          f"{100 * macs['quantized_share_dev']:.1f}% of them run in quantized Linear layers")

    # ---------------------------------------------------------------- save
    results = {
        "method": args.method,
        "description": METHOD_DESCRIPTIONS[args.method],
        "settings": {
            "device": "cpu", "threads": torch.get_num_threads(), "quantized_engine": engine,
            "attn_implementation": attn_impl, "batch_size": config.BATCH_SIZE,
            "max_length": config.MAX_LENGTH, "warmup_iters": config.WARMUP_ITERS,
            "batched_repeats": config.BATCHED_REPEATS, "limit": args.limit,
        },
        "data": data_info,
        "layers": {k: len(v) for k, v in layers.items()},
        "quantization_time_s": quantization_time_s,
        "params": params,
        "macs": macs,
        "memory": {
            "serialized_size_mb": size_mb,
            "rss_model_mb": rss_after_model - rss_before_model,     # RAM taken by the loaded model
            "rss_peak_process_mb": monitor.peak_mb,                 # whole process, peak during inference
            "rss_peak_inference_extra_mb": monitor.peak_mb - rss_ready,  # extra RAM used while running
        },
        "accuracy": metrics,
        "latency_bs1": lat_bs1,
        "latency_batched": lat_batched,
        "environment": collect_environment(),
    }

    out_path = os.path.join(args.results_dir, f"{args.method}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.results_dir, "preds", f"{args.method}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"preds": preds.tolist(), "labels": gold.tolist()}, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
