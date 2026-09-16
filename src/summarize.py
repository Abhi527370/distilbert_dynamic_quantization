"""
summarize.py  -  collect results/*.json into report-ready tables and figures.

    python -m src.summarize

Creates in results/:
    summary_full.csv      every number, one row per method
    summary.md            two compact Markdown tables (copy them into the report)
    fig_latency.png       latency comparison
    fig_size_memory.png   model size and peak RAM
    fig_tradeoff.png      accuracy vs latency
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

SHORT_NAMES = {
    "fp32": "FP32 (baseline)",
    "dq_int8_per_tensor": "INT8 per-tensor",
    "dq_int8_per_channel": "INT8 per-channel",
    "dq_int8_linear_emb": "INT8 per-ch. + 8-bit emb.",
    "dq_fp16": "FP16 weights",
}


def mcnemar_exact_p(b, c):
    """
    Exact two-sided McNemar test: are the two models' errors really different,
    or could the accuracy gap be chance? b, c = sentences only one model got right.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_table(results_dir):
    # ALL_METHODS = fixed display order; only the files that actually exist are used
    results = {m: load_json(os.path.join(results_dir, f"{m}.json"))
               for m in config.ALL_METHODS if os.path.exists(os.path.join(results_dir, f"{m}.json"))}
    if not results:
        raise FileNotFoundError(f"No result files in {results_dir}. Run run_all.py first.")
    base = results.get("fp32")

    def load_preds(method):
        path = os.path.join(results_dir, "preds", f"{method}.json")
        return load_json(path) if os.path.exists(path) else None

    base_preds = load_preds("fp32")
    rows = []
    for method, r in results.items():
        by = r["params"]["by_storage_dtype"]
        row = {
            "method": method,
            "name": SHORT_NAMES.get(method, method),
            "params_total_M": r["params"]["total"] / 1e6,
            "params_fp32_M": by.get("float32", 0) / 1e6,
            "params_int8_M": (by.get("qint8", 0) + by.get("quint8", 0)) / 1e6,
            "params_fp16_M": by.get("float16", 0) / 1e6,
            "gmacs_dev": r["macs"]["per_sentence_dev_mean"] / 1e9,
            "gmacs_128": r["macs"][f"per_sentence_{config.MAC_REFERENCE_LENGTH}_tokens"] / 1e9,
            "macs_quantized_pct": 100 * r["macs"]["quantized_share_dev"],
            "weights_theory_mb": r["params"]["raw_weight_bytes_mb"],
            "size_mb": r["memory"]["serialized_size_mb"],
            "rss_model_mb": r["memory"]["rss_model_mb"],
            "rss_peak_mb": r["memory"]["rss_peak_process_mb"],
            "lat_bs1_mean_ms": r["latency_bs1"]["mean_ms"],
            "lat_bs1_std_ms": r["latency_bs1"]["std_ms"],
            "lat_bs1_median_ms": r["latency_bs1"]["median_ms"],
            "lat_bs1_p95_ms": r["latency_bs1"]["p95_ms"],
            "batched_ms_per_sample": r["latency_batched"]["ms_per_sample"],
            "accuracy_pct": 100 * r["accuracy"]["accuracy"],
            "macro_f1": r["accuracy"]["macro_f1"],
            "quant_time_s": r["quantization_time_s"],
        }
        if base is not None:
            row["size_reduction_x"] = base["memory"]["serialized_size_mb"] / row["size_mb"]
            row["speedup_bs1"] = base["latency_bs1"]["median_ms"] / row["lat_bs1_median_ms"]
            row["speedup_batched"] = base["latency_batched"]["ms_per_sample"] / row["batched_ms_per_sample"]
            row["delta_acc_pp"] = row["accuracy_pct"] - 100 * base["accuracy"]["accuracy"]
        preds = load_preds(method)
        if base_preds is not None and preds is not None:
            p, p0 = np.array(preds["preds"]), np.array(base_preds["preds"])
            y = np.array(preds["labels"])
            fixed = int(np.sum((p0 != y) & (p == y)))   # FP32 wrong -> now right
            broken = int(np.sum((p0 == y) & (p != y)))  # FP32 right -> now wrong
            row.update({"agreement_pct": 100 * float(np.mean(p == p0)),
                        "fixed": fixed, "broken": broken,
                        "mcnemar_p": mcnemar_exact_p(fixed, broken)})
        rows.append(row)
    return pd.DataFrame(rows), results


def markdown_table(df, columns):
    """columns = list of (header, function(row) -> str)."""
    lines = ["| " + " | ".join(h for h, _ in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fn(row) for _, fn in columns) + " |")
    return "\n".join(lines)


def _get(row, key, fmt, default="-"):
    value = row.get(key, None)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return fmt.format(value)


def write_markdown(df, results, results_dir):
    table_cost = markdown_table(df, [
        ("Method", lambda r: r["name"]),
        ("Params (M)", lambda r: f"{r['params_total_M']:.2f}"),
        ("FP32 / INT8 / FP16 params (M)",
         lambda r: f"{r['params_fp32_M']:.1f} / {r['params_int8_M']:.1f} / {r['params_fp16_M']:.1f}"),
        ("GMACs / sentence (dev avg)", lambda r: f"{r['gmacs_dev']:.3f}"),
        (f"GMACs @{config.MAC_REFERENCE_LENGTH} tok", lambda r: f"{r['gmacs_128']:.3f}"),
        ("% MACs in quantized layers", lambda r: f"{r['macs_quantized_pct']:.1f}"),
    ])
    table_results = markdown_table(df, [
        ("Method", lambda r: r["name"]),
        ("Weights MB (theory)", lambda r: f"{r['weights_theory_mb']:.1f}"),
        ("Saved size MB (x smaller)", lambda r: f"{r['size_mb']:.1f} ({_get(r, 'size_reduction_x', '{:.2f}')}x)"),
        ("Model RAM MB", lambda r: f"{r['rss_model_mb']:.0f}"),
        ("Peak RAM MB", lambda r: f"{r['rss_peak_mb']:.0f}"),
        ("Latency bs=1 mean+/-std ms", lambda r: f"{r['lat_bs1_mean_ms']:.2f} +/- {r['lat_bs1_std_ms']:.2f}"),
        ("bs=1 median ms (speedup)",
         lambda r: f"{r['lat_bs1_median_ms']:.2f} ({_get(r, 'speedup_bs1', '{:.2f}')}x)"),
        (f"bs={config.BATCH_SIZE} ms/sent. (speedup)",
         lambda r: f"{r['batched_ms_per_sample']:.2f} ({_get(r, 'speedup_batched', '{:.2f}')}x)"),
        ("Accuracy % (delta pp)", lambda r: f"{r['accuracy_pct']:.2f} ({_get(r, 'delta_acc_pp', '{:+.2f}')})"),
        ("Macro-F1", lambda r: f"{r['macro_f1']:.4f}"),
        ("Agree w/ FP32 %", lambda r: _get(r, "agreement_pct", "{:.2f}")),
        ("fixed / broken", lambda r: f"{_get(r, 'fixed', '{:.0f}')} / {_get(r, 'broken', '{:.0f}')}"),
        ("McNemar p", lambda r: _get(r, "mcnemar_p", "{:.3f}")),
    ])

    env = next(iter(results.values()))["environment"]
    settings = next(iter(results.values()))["settings"]
    lines = [
        "# Results summary", "",
        f"- CPU: {env['cpu_model']} | {env['cpu_physical_cores']} physical / "
        f"{env['cpu_logical_cores']} logical cores | RAM {env['ram_total_gb']} GB | "
        f"ISA: {env.get('torch_cpu_capability')}",
        f"- Threads used: {settings['threads']} | quantized engine: {settings['quantized_engine']} | "
        f"attention: {settings['attn_implementation']}",
        f"- Python {env['python']} | torch {env['torch']} | transformers {env['transformers']} | "
        f"datasets {env['datasets']}",
        f"- Dev set: {next(iter(results.values()))['data']['n_sentences']} sentences",
        "",
        "## Model cost (parameters and MACs)", "", table_cost, "",
        "## Memory, latency and accuracy (all on CPU)", "", table_results, "",
        "Notes: speedups use the median bs=1 latency and the batched ms/sentence. "
        "'fixed' = sentences FP32 got wrong and this model got right; 'broken' = the opposite. "
        "McNemar p > 0.05 means the accuracy difference to FP32 is not statistically significant.",
    ]
    path = os.path.join(results_dir, "summary.md")
    # encoding="utf-8" matters on Windows, whose default (cp1252) cannot
    # encode many characters and would crash here.
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def make_figures(df, results_dir):
    names = df["name"].tolist()
    x = np.arange(len(names))

    # 1) latency
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(x, df["lat_bs1_median_ms"], color="tab:blue")
    axes[0].set_title("Batch size 1: median latency per sentence")
    axes[0].set_ylabel("ms")
    axes[1].bar(x, df["batched_ms_per_sample"], color="tab:orange")
    axes[1].set_title(f"Batch size {config.BATCH_SIZE}: time per sentence")
    axes[1].set_ylabel("ms")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fig_latency.png"), dpi=200)
    plt.close(fig)

    # 2) size + peak RAM
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.38
    ax.bar(x - width / 2, df["size_mb"], width, label="saved model size")
    ax.bar(x + width / 2, df["rss_peak_mb"], width, label="peak process RAM (inference)")
    ax.set_ylabel("MB")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fig_size_memory.png"), dpi=200)
    plt.close(fig)

    # 3) accuracy vs latency trade-off (one colour/marker per method)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    for i, (_, r) in enumerate(df.iterrows()):
        ax.scatter(r["lat_bs1_median_ms"], r["accuracy_pct"], s=70, alpha=0.8,
                   marker=markers[i % len(markers)],
                   label=f"{r['name']} ({r['size_mb']:.0f} MB)")
    ax.set_xlabel("median latency, batch size 1 (ms)  - lower is better")
    ax.set_ylabel("SST-2 dev accuracy (%)")
    ax.set_title("Accuracy vs latency (legend: saved model size)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fig_tradeoff.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--results_dir", default=config.RESULTS_DIR)
    args = parser.parse_args()

    df, results = build_table(args.results_dir)
    df.to_csv(os.path.join(args.results_dir, "summary_full.csv"), index=False)
    md_path = write_markdown(df, results, args.results_dir)
    make_figures(df, args.results_dir)

    print(open(md_path, encoding="utf-8").read())
    print(f"Saved summary_full.csv, summary.md and 3 figures in {args.results_dir}/")


if __name__ == "__main__":
    main()
