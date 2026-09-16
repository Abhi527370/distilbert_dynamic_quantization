"""
config.py  -  ALL experiment settings live in this one file.
"""

# ---------------------------------------------------------------------------
# DistilBERT-base-uncased, already fine-tuned on SST-2 by Hugging Face.
# https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# SST-2 (Stanford Sentiment Treebank, binary sentiment: 0 = negative, 1 = positive)
# https://huggingface.co/datasets/stanfordnlp/sst2
# Splits: train 67,349 | validation 872 | test 1,821.
# The TEST labels are hidden (all = -1), so accuracy can only be computed on the
# VALIDATION split (= the "dev" set). This is the standard GLUE practice.
DATASET_NAME = "stanfordnlp/sst2"
EVAL_SPLIT = "validation"
TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"

# SST-2 sentences are short (< 70 tokens), so nothing is cut off at 128.
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Measurement settings
# ---------------------------------------------------------------------------
# PyTorch CPU threads. 1 thread is slower in absolute terms but gives the most
# STABLE timings (which is what matters when comparing models)
# Override it with:  python run_all.py --threads 4
NUM_THREADS = 1

BATCH_SIZE = 32            # batch size for accuracy + batched-latency measurement
WARMUP_ITERS = 20          # untimed forward passes before batch-size-1 timing
BATCHED_WARMUP_ITERS = 3   # untimed batches before batched timing
BATCHED_REPEATS = 3        # the batched loop over the dev set is timed this many times
MAC_REFERENCE_LENGTH = 128 # MACs are also reported for one input of exactly 128 tokens
MEMORY_POLL_INTERVAL_S = 0.001  # the peak-RAM monitor samples memory every 1 ms
SEED = 0

RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Models to benchmark, in this order (definitions are in src/quantization.py)
# ---------------------------------------------------------------------------
# DEFAULT = the FP32 baseline + exactly 2 quantized variants. These two were chosen
# because they differ on every axis (precision, size, speed, accuracy)
#   - dq_int8_per_tensor : INT8, the standard recipe -> smaller AND faster
#   - dq_fp16            : FP16, same API, different dtype -> smaller but NOT faster
METHODS = [
    "fp32",                 # baseline, no quantization
    "dq_int8_per_tensor",   # (1) INT8 Linear, 1 scale per weight matrix
    "dq_fp16",              # (2) FP16 Linear weights
]

#   python run_all.py --methods fp32 dq_int8_per_tensor dq_fp16 dq_int8_per_channel
#   - dq_int8_per_channel : INT8 Linear, 1 scale per output neuron (finer scales)
#   - dq_int8_linear_emb  : per-channel INT8 Linear + 8-bit embeddings (much smaller)
ALL_METHODS = [
    "fp32",
    "dq_int8_per_tensor",
    "dq_int8_per_channel",
    "dq_int8_linear_emb",
    "dq_fp16",
]
