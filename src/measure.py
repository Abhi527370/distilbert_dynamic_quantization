"""
measure.py  -  helper functions that measure one model:
    * number of parameters (and in which precision they are stored)
    * model size (MB) and RAM usage (MB)
    * latency (ms)
    * accuracy / macro-F1
MAC counting lives in src/macs.py.
"""
import ctypes
import gc
import io
import os
import threading
import time

import numpy as np
import psutil
import torch

from src.quantization import nnq, nnqd

MB = 1e6  # all sizes are reported in megabytes (10^6 bytes)
_PROCESS = psutil.Process(os.getpid())


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def rss_mb():
    """Resident Set Size = RAM currently used by this Python process (MB)."""
    return _PROCESS.memory_info().rss / MB


def release_memory():
    """Free unused Python objects and give free C memory back to the OS,
    so that the next RSS reading reflects what is really still in use."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux (glibc) only
    except (OSError, AttributeError):
        pass


class PeakRSSMonitor:
    """
    Context manager that samples the process RAM every `interval` seconds in a
    background thread and remembers the highest value (= peak RAM).

        with PeakRSSMonitor() as monitor:
            run_inference()
        print(monitor.peak_mb)
    """

    def __init__(self, interval=0.001):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_mb = rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb())
        return False


def serialized_size_mb(model):
    """Size of the saved state_dict (same method as the PyTorch tutorial)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size = buffer.getbuffer().nbytes / MB
    del buffer
    return size


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
_BYTES_PER_ELEMENT = {"float32": 4, "float16": 2, "qint8": 1, "quint8": 1}


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def count_parameters(model):
    """
    Count ALL parameters, grouped by the precision in which they are stored.

    Quantized layers keep their weights in special "packed" objects that
    model.parameters() does NOT return, so we unpack them explicitly.
    Quantization never changes the NUMBER of parameters - only their bits.
    """
    counts = {}

    def add(dtype_name, n):
        counts[dtype_name] = counts.get(dtype_name, 0) + int(n)

    # 1) ordinary float parameters (LayerNorm, biases, non-quantized layers)
    for p in model.parameters():
        add(_dtype_name(p.dtype), p.numel())

    # 2) weights hidden inside quantized modules
    for module in model.modules():
        if isinstance(module, nnqd.Linear):
            weight, bias = module._weight_bias()
            add(_dtype_name(module._packed_params.dtype), weight.numel())  # qint8 or float16
            if bias is not None:
                add("float32", bias.numel())  # biases stay FP32
        elif isinstance(module, nnq.Embedding):
            add(_dtype_name(module._packed_params.dtype), module.weight().numel())  # quint8

    total = sum(counts.values())
    weight_mb = sum(n * _BYTES_PER_ELEMENT.get(k, 4) for k, n in counts.items()) / MB
    return {"total": total, "by_storage_dtype": counts, "raw_weight_bytes_mb": weight_mb}


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def classification_metrics(preds, labels):
    """Accuracy (the official SST-2 / GLUE metric) + macro-F1 as a secondary metric."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    f1_per_class = []
    for c in (0, 1):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        denom = 2 * tp + fp + fn
        f1_per_class.append(2 * tp / denom if denom > 0 else 0.0)
    return {
        "accuracy": float(np.mean(preds == labels)),
        "macro_f1": float(np.mean(f1_per_class)),
        "n_correct": int(np.sum(preds == labels)),
        "n_samples": int(len(labels)),
    }


@torch.inference_mode()
def evaluate_accuracy(model, batches):
    """Run the model over all dev batches (CPU). Returns (metrics, predictions, labels)."""
    all_preds, all_labels = [], []
    for inputs, labels in batches:
        logits = model(**inputs).logits
        all_preds.append(logits.argmax(dim=-1))
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return classification_metrics(preds, labels), preds, labels


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def _sync(device):
    """GPU work is asynchronous; wait for it before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_stats(times_ms):
    t = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "min_ms": float(t.min()),
        "n_runs": int(t.size),
    }


@torch.inference_mode()
def latency_single(model, single_inputs, warmup, device=torch.device("cpu")):
    """
    Batch size 1: time one forward pass per dev sentence (real lengths, no padding).
    Warm-up passes are run first and not timed (first calls are always slower).
    """
    inputs = [{k: v.to(device) for k, v in x.items()} for x in single_inputs]
    for i in range(warmup):
        model(**inputs[i % len(inputs)])
    _sync(device)

    times_ms = []
    for x in inputs:
        _sync(device)
        start = time.perf_counter()
        model(**x)
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return _time_stats(times_ms)


@torch.inference_mode()
def latency_batched(model, batches, warmup, repeats, device=torch.device("cpu")):
    """
    Batched inference: time every batch of the dev set, `repeats` times.
    Reports ms per batch and ms per sentence (= total time / total sentences).
    """
    inputs = [{k: v.to(device) for k, v in x.items()} for x, _ in batches]
    n_sentences = sum(x["input_ids"].shape[0] for x in inputs)
    for i in range(warmup):
        model(**inputs[i % len(inputs)])
    _sync(device)

    times_ms = []
    for _ in range(repeats):
        for x in inputs:
            _sync(device)
            start = time.perf_counter()
            model(**x)
            _sync(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    total_ms = float(np.sum(times_ms))
    per_batch = _time_stats(times_ms)
    return {
        "batch_size": int(inputs[0]["input_ids"].shape[0]),
        "ms_per_batch_mean": per_batch["mean_ms"],
        "ms_per_batch_median": per_batch["median_ms"],
        "ms_per_sample": total_ms / (n_sentences * repeats),
        "throughput_samples_per_s": 1000.0 * n_sentences * repeats / total_ms,
        "repeats": repeats,
    }
