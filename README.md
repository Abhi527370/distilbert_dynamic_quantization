# Dynamic Quantization of DistilBERT

A small, reproducible benchmark that measures how PyTorch **dynamic quantization**
changes the inference efficiency of a DistilBERT sentiment classifier. For an FP32
baseline and one or more quantized variants it reports, on the CPU: parameter count,
MAC operations, latency, memory use, and accuracy — with figures and report-ready
tables generated automatically.

- **Model:** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- **Dataset:** [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), validation split (872 sentences). The GLUE test labels are withheld, so accuracy is computed on the validation split.
- **Stack:** PyTorch (`torch.ao.quantization.quantize_dynamic`), Hugging Face `transformers` and `datasets`.

Everything runs on the CPU — PyTorch dynamic quantization has no GPU kernels, so the
FP32 baseline is measured on the CPU too for a fair comparison. No GPU is required and
the whole run stays under ~2 GB of RAM.

## Quick start

Python 3.9+ is required. No internet is needed after the first download.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python check_setup.py       # verify which methods this machine supports (~5 s)
python run_all.py --smoke   # quick end-to-end check on 64 sentences (~2 min)
python run_all.py           # full benchmark (~10–25 min, 1 thread)
```

`check_setup.py` probes each quantization method on a tiny model and prints a command
that is guaranteed to work on the current machine; run it, and the smoke test, before
the full benchmark. The first run downloads the model (~260 MB) and dataset (~7 MB)
and caches them, so later runs work offline.

## Quantization methods

All methods use `quantize_dynamic`: linear-layer weights are converted to low precision
ahead of time, activations are quantized on the fly from their observed range (so no
calibration data is needed), and the attention score matmuls, softmax, GELU and
LayerNorm remain FP32.

| Method | What is quantized |
|---|---|
| `dq_int8_per_tensor` | all `nn.Linear`, INT8, one weight scale per matrix |
| `dq_int8_per_channel` | all `nn.Linear`, INT8, one weight scale per output channel |
| `dq_int8_linear_emb` | per-channel INT8 `nn.Linear` + 8-bit embedding tables |
| `dq_fp16` | all `nn.Linear`, weights stored in FP16 (FBGEMM backend only) |

Select any subset with `--methods`, always including `fp32` as the baseline:

```bash
python run_all.py --methods fp32 dq_int8_per_tensor dq_int8_linear_emb
```

> **Backend note.** INT8 runs on any CPU backend (FBGEMM, oneDNN, x86, QNNPACK). FP16
> dynamic quantization is implemented only in FBGEMM; on a build without it, `dq_fp16`
> is skipped and `check_setup.py` suggests an alternative. A missing FBGEMM is a
> property of the PyTorch build, not of the CPU.

## What is measured

Each model is benchmarked in a **fresh process** (so memory readings never mix) with a
single thread (more stable for comparison). Tokenization happens before timing, so
latency reflects only the forward pass.

- **Parameters** — counted including weights packed inside quantized modules, grouped by storage precision. Quantization changes the bits per parameter, not the parameter count.
- **MACs** — via `torch.utils.flop_counter.FlopCounterMode` (MACs = FLOPs / 2), cross-checked against a hand-derived formula, reported per sentence and for a 128-token input.
- **Latency** — batch size 1 (per-sentence, after warm-up: mean, std, median, p95) and batch size 32 (throughput).
- **Memory** — theoretical weight bytes, serialized `state_dict` size, and peak process RSS during inference.
- **Accuracy** — accuracy and macro-F1, plus per-model agreement with the baseline and an exact McNemar test to check whether accuracy differences are statistically significant.

## Project structure

```
├── check_setup.py     # reports which methods the local backend supports
├── run_all.py         # runs the full pipeline end to end
├── requirements.txt
└── src/
    ├── config.py       # all settings: model, dataset, threads, batch size, methods
    ├── data.py         # load and tokenize the dataset
    ├── quantization.py # load the FP32 model and build the quantized variants
    ├── measure.py      # parameters, size, RAM, latency, accuracy
    ├── macs.py         # MAC counting and formula cross-check
    ├── benchmark.py    # measure one model -> results/<method>.json
    ├── environment.py  # hardware and library versions
    └── summarize.py    # aggregate results into tables and figures
```

## Output

Results are written to `results/`:

| File | Content |
|---|---|
| `summary.md`, `summary_full.csv` | report-ready comparison tables |
| `fig_latency.png`, `fig_size_memory.png`, `fig_tradeoff.png` | comparison figures |
| `<method>.json` | all raw measurements for one model |
| `preds/<method>.json` | per-sentence predictions (agreement / McNemar) |
| `macs.json`, `environment.json` | MAC counts and the full environment record |

## Useful options

```bash
python run_all.py --threads 4                         # use 4 CPU threads
python run_all.py --results_dir results_run2          # write to a separate folder
python -m src.benchmark --method dq_int8_per_tensor   # benchmark a single method
python -m src.summarize                               # rebuild tables and figures
```

For trustworthy latency, close other heavy programs, keep a laptop on mains power, and
leave the machine idle while it times. Latency is hardware-specific; the ratios between
models are what transfer.

## Configuration

Every setting — model, dataset, thread count, batch size, warm-up iterations, and the
default method list — lives in `src/config.py`. No other file contains hard-coded
settings.
