"""
quantization.py  -  load the FP32 model and build the dynamically quantized variants.

What does PyTorch *dynamic* quantization (`quantize_dynamic`) do?
  * The WEIGHTS of the selected layer types are converted once, ahead of time,
    to lower precision (INT8 or FP16) and stored in that format.
  * The ACTIVATIONS (inputs of those layers) stay FP32 in memory. At run time,
    just before every quantized nn.Linear, the input is quantized to 8 bit
    using its *current* min/max values - this is the "dynamic" part, so no
    calibration data is needed. The matrix multiply then runs as
    INT8 x INT8 -> INT32, and the result is converted back to FP32.
  * All other operations (Q.K^T and A.V attention matmuls, softmax, GELU,
    LayerNorm) keep running in FP32.
  * PyTorch's kernels for this are CPU-only (FBGEMM on x86, QNNPACK on ARM),
    so every model in this project is measured on the CPU.
"""
import warnings

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from src import config

warnings.filterwarnings("ignore", message=r".*torch\.ao\.quantization.*")

try:
    from torch.ao.quantization import (
        default_dynamic_qconfig,            # INT8 weights, one scale per tensor
        float16_dynamic_qconfig,            # FP16 weights
        float_qparams_weight_only_qconfig,  # 8-bit embeddings, one scale per row
        per_channel_dynamic_qconfig,        # INT8 weights, one scale per output channel
        quantize_dynamic,
    )
    import torch.ao.nn.quantized as nnq            # quantized Embedding lives here
    import torch.ao.nn.quantized.dynamic as nnqd   # dynamic quantized Linear lives here
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "This PyTorch version has no torch.ao.quantization. Install a version that "
        "still has it, e.g.:  pip install torch==2.5.1 --index-url "
        "https://download.pytorch.org/whl/cpu"
    ) from err


# Human-readable description of every method (also written into the results).
METHOD_DESCRIPTIONS = {
    "fp32": "FP32 baseline, no quantization",
    "dq_int8_per_tensor": "Dynamic INT8 on all nn.Linear; one weight scale per matrix (per-tensor, symmetric)",
    "dq_int8_per_channel": "Dynamic INT8 on all nn.Linear; one weight scale per output channel (per-channel, symmetric)",
    "dq_int8_linear_emb": "Per-channel dynamic INT8 on all nn.Linear + 8-bit weight-only nn.Embedding (row-wise scales)",
    "dq_fp16": "Dynamic FP16 on all nn.Linear (weights stored in float16)",
}

# Which layer type gets which quantization config ("qconfig") for each method.
QCONFIG_SPECS = {
    # (1) The standard recipe from the PyTorch BERT tutorial.
    "dq_int8_per_tensor": {nn.Linear: default_dynamic_qconfig},
    # (2) Same, but a separate scale for every output neuron. Rows of a weight
    #     matrix can have very different ranges; per-channel scales follow them
    #     more closely -> smaller rounding error, same size and speed.
    "dq_int8_per_channel": {nn.Linear: per_channel_dynamic_qconfig},
    # (3) (2) + the embedding tables. The word-embedding table alone is 23.4M of
    #     the 67M parameters and stays FP32 in (1)/(2). Embeddings are look-ups
    #     (no MACs), so this should cut model size but not latency.
    "dq_int8_linear_emb": {
        nn.Linear: per_channel_dynamic_qconfig,
        nn.Embedding: float_qparams_weight_only_qconfig,
    },
    # (4) Half-precision weights: half the size of FP32 Linear weights, almost
    #     no accuracy risk, but the CPU still computes in FP32, so little/no speedup is expected.
    "dq_fp16": {nn.Linear: float16_dynamic_qconfig},
}


# Which CPU backend runs the quantized kernels, best first.
#   x86     = modern wrapper that picks FBGEMM or oneDNN automatically
#   fbgemm  = classic x86 backend; ONLY registers if the CPU has AVX2
#   onednn  = Intel oneDNN; works without AVX2 (the usual fallback)
#   qnnpack = ARM backend
ENGINE_PREFERENCE = ("x86", "fbgemm", "onednn", "qnnpack")


def available_engines():
    """Quantized engines this PyTorch build + this CPU actually support."""
    return [e for e in torch.backends.quantized.supported_engines if e != "none"]


def select_quantized_engine():
    """Pick the best available CPU backend for the quantized kernels."""
    engines = available_engines()
    for engine in ENGINE_PREFERENCE:
        if engine in engines:
            torch.backends.quantized.engine = engine
            return engine
    if engines:  # something unexpected but usable - take it rather than crash
        torch.backends.quantized.engine = engines[0]
        return engines[0]
    raise RuntimeError(
        "This PyTorch build reports no CPU quantized engine at all "
        f"(supported_engines = {torch.backends.quantized.supported_engines}). "
        "Reinstall the official CPU wheel: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )


class _TinyModel(nn.Module):
    """Smallest model containing both layer types, used only to test support."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.fc = nn.Linear(8, 4)

    def forward(self, ids):
        return self.fc(self.emb(ids))


def probe_method(method):
    """
    Try one quantization method on a tiny model to see whether this machine
    supports it. Returns (True, "") or (False, "reason").

    Not every method exists on every CPU. In particular FP16 dynamic
    quantization needs FBGEMM, which needs AVX2.
    """
    if method == "fp32":
        return True, ""
    if method not in QCONFIG_SPECS:
        return False, f"unknown method '{method}'"
    try:
        model = _TinyModel().eval()
        quantized = quantize_dynamic(model, qconfig_spec=QCONFIG_SPECS[method], inplace=False)
        with torch.inference_mode():
            quantized(torch.randint(0, 16, (2, 3)))
        return True, ""
    except Exception as err:
        reason = " ".join(str(err).split())
        return False, reason[:220]


def load_fp32_model(attn_implementation=None):
    """
    Load the pretrained, SST-2 fine-tuned DistilBERT in FP32 and eval mode.
    attn_implementation=None -> the library default (usually "sdpa").
    We use "eager" only for MAC counting (plain matmuls the counter can see).
    """
    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, **kwargs)
    model = model.float()  # make sure all weights are FP32
    model.eval()           # turn dropout off (inference mode)
    return model


def check_label_mapping(model):
    """SST-2 uses 0 = negative, 1 = positive. Make sure the model agrees."""
    id2label = {int(k): str(v).upper() for k, v in model.config.id2label.items()}
    if not (id2label.get(0, "").startswith("NEG") and id2label.get(1, "").startswith("POS")):
        raise ValueError(f"Unexpected label mapping {id2label}; expected 0=NEGATIVE, 1=POSITIVE.")


def apply_quantization(model_fp32, method):
    """
    Return a NEW quantized copy of `model_fp32` (the original is not changed).
    For method == "fp32" the model is returned unchanged.
    """
    if method == "fp32":
        return model_fp32
    if method not in QCONFIG_SPECS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHOD_DESCRIPTIONS)}")

    model_fp32.eval()
    quantized = quantize_dynamic(model_fp32, qconfig_spec=QCONFIG_SPECS[method], inplace=False)
    _sanity_check(quantized, method)
    return quantized


def describe_layers(model):
    """Names of quantized / non-quantized Linear and Embedding layers."""
    info = {"quantized_linear": [], "float_linear": [],
            "quantized_embedding": [], "float_embedding": []}
    for name, module in model.named_modules():
        if isinstance(module, nnqd.Linear):
            info["quantized_linear"].append(name)
        elif isinstance(module, nnq.Embedding):
            info["quantized_embedding"].append(name)
        elif isinstance(module, nn.Linear):
            info["float_linear"].append(name)
        elif isinstance(module, nn.Embedding):
            info["float_embedding"].append(name)
    return info


def _sanity_check(model, method):
    """Fail loudly if quantization silently did not happen."""
    layers = describe_layers(model)
    if layers["float_linear"] or not layers["quantized_linear"]:
        raise RuntimeError(f"{method}: some nn.Linear layers were not quantized: {layers['float_linear']}")
    if method == "dq_int8_linear_emb" and layers["float_embedding"]:
        raise RuntimeError(f"{method}: embeddings were not quantized: {layers['float_embedding']}")
    print(f"  quantized {len(layers['quantized_linear'])} nn.Linear layers"
          f" and {len(layers['quantized_embedding'])} nn.Embedding layers")
