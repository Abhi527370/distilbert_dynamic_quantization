"""
data.py  -  load the SST-2 dev set and turn the sentences into model inputs.

Tokenization is done ONCE, before any timing starts, so the latency numbers
measure only the neural-network forward pass (not the tokenizer).
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src import config


def load_tokenizer():
    """The tokenizer that belongs to the fine-tuned DistilBERT checkpoint."""
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


def load_sst2_dev(limit=None):
    """
    Return (sentences, labels) of the SST-2 validation (dev) split.
    `limit` = use only the first N sentences (for a quick smoke test).
    """
    dataset = load_dataset(config.DATASET_NAME, split=config.EVAL_SPLIT)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    sentences = [str(s) for s in dataset[config.TEXT_COLUMN]]
    labels = [int(y) for y in dataset[config.LABEL_COLUMN]]

    if any(y not in (0, 1) for y in labels):
        raise ValueError(
            "Found labels other than 0/1. The SST-2 *test* split has hidden "
            "labels (-1) - use the validation split for evaluation."
        )
    return sentences, labels


def _model_inputs(encoding):
    """DistilBERT only needs input_ids and attention_mask (no token_type_ids)."""
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }


def make_single_inputs(tokenizer, sentences):
    """
    One input per sentence (batch size 1, no padding).
    Used for per-sentence latency, i.e. the "one user request" scenario.
    """
    return [
        _model_inputs(
            tokenizer(s, truncation=True, max_length=config.MAX_LENGTH,
                      return_tensors="pt")
        )
        for s in sentences
    ]


def make_batches(tokenizer, sentences, labels, batch_size):
    """
    List of (inputs, labels) batches. Each batch is padded only up to the
    longest sentence in that batch ("dynamic padding"). The attention mask
    makes padding tokens invisible to the model, so padding does not change
    the predictions.
    """
    batches = []
    for start in range(0, len(sentences), batch_size):
        encoding = tokenizer(
            sentences[start:start + batch_size],
            padding=True, truncation=True, max_length=config.MAX_LENGTH,
            return_tensors="pt",
        )
        y = torch.tensor(labels[start:start + batch_size], dtype=torch.long)
        batches.append((_model_inputs(encoding), y))
    return batches


def length_stats(single_inputs):
    """Token-length statistics of the dev set (reported in the results)."""
    lengths = [x["input_ids"].shape[1] for x in single_inputs]
    return {
        "n_sentences": len(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        # sentences that reached MAX_LENGTH (0 expected for SST-2)
        "n_at_max_length": sum(1 for n in lengths if n >= config.MAX_LENGTH),
    }
