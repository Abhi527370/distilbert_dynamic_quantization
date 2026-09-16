# Results summary

- CPU: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz | 4 physical / 8 logical cores | RAM 16.86 GB | ISA: AVX512
- Threads used: 1 | quantized engine: onednn | attention: sdpa
- Python 3.12.2 | torch 2.14.0+cpu | transformers 5.17.0 | datasets 5.0.1
- Dev set: 872 sentences

## Model cost (parameters and MACs)

| Method | Params (M) | FP32 / INT8 / FP16 params (M) | GMACs / sentence (dev avg) | GMACs @128 tok | % MACs in quantized layers |
|---|---|---|---|---|---|
| FP32 (baseline) | 66.96 | 67.0 / 0.0 / 0.0 | 1.076 | 5.587 | 0.0 |
| INT8 per-tensor | 66.96 | 23.9 / 43.1 / 0.0 | 1.076 | 5.587 | 99.4 |
| INT8 per-ch. + 8-bit emb. | 66.96 | 0.1 / 66.9 / 0.0 | 1.076 | 5.587 | 99.4 |

## Memory, latency and accuracy (all on CPU)

| Method | Weights MB (theory) | Saved size MB (x smaller) | Model RAM MB | Peak RAM MB | Latency bs=1 mean+/-std ms | bs=1 median ms (speedup) | bs=32 ms/sent. (speedup) | Accuracy % (delta pp) | Macro-F1 | Agree w/ FP32 % | fixed / broken | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 (baseline) | 267.8 | 267.9 (1.00x) | 271 | 774 | 44.25 +/- 10.25 | 42.24 (1.00x) | 51.79 (1.00x) | 91.06 (+0.00) | 0.9104 | 100.00 | 0 / 0 | 1.000 |
| INT8 per-tensor | 138.6 | 138.7 (1.93x) | 290 | 828 | 14.15 +/- 3.35 | 13.63 (3.10x) | 17.53 (2.95x) | 90.02 (-1.03) | 0.9000 | 95.76 | 14 / 23 | 0.188 |
| INT8 per-ch. + 8-bit emb. | 67.1 | 68.2 (3.93x) | 218 | 739 | 15.59 +/- 3.53 | 15.00 (2.82x) | 18.59 (2.79x) | 90.14 (-0.92) | 0.9010 | 96.10 | 13 / 21 | 0.229 |

Notes: speedups use the median bs=1 latency and the batched ms/sentence. 'fixed' = sentences FP32 got wrong and this model got right; 'broken' = the opposite. McNemar p > 0.05 means the accuracy difference to FP32 is not statistically significant.
