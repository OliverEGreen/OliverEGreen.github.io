"""
Trains a decoder-only (GPT-style) transformer on BPE subword tokens. Aggressively
optimised NumPy port of raw_transformers.py, tuned for Apple Silicon with float32 arrays,
cached masks, random corpus windows, and low-allocation Adam updates. Byte-pair-encoding
tokenisation: words are split into learned subword pieces, with a "</w>" boundary token
between words so spacing survives. The tokeniser is trained from the corpus at startup.

Mirror of basic_lstm_numpy.py's structure, expanded for the transformer's stack of
blocks (multi-head causal self-attention + position-wise feed-forward, each wrapped in
add-&-norm). Post-layer-norm and learned positional embeddings, exactly as in the raw
version.

Shape conventions: a sequence is processed as (T, EMBEDDING_DIM) — one row per position,
the "belt". Attention batches the heads as (NUM_HEADS, T, HEAD_DIM). Every weight is
stored (out, in) so a projection is `x @ W.T`, matching the raw file's matrix_vector
convention.
"""

import argparse
import hashlib
import json
import pickle
import re
import time
from pathlib import Path
from statistics import mean

import numpy as np

# M1/Accelerate is very fast at float32 matrix math; float64 roughly doubles memory
# bandwidth for no useful gain in this character model.
DTYPE = np.float32

# Quality/performance defaults tuned for Apple Silicon NumPy.
EMBEDDING_DIM = 256
NUM_HEADS = 8
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
FF_HIDDEN_DIM = 1024
LAYER_NORM_EPS = 1e-5
NUM_LAYERS = 6
SEQUENCE_LENGTH = 256
SAMPLE_SIZE = 600

GRADIENT_CLIP = 1.0
LEARNING_RATE = 7e-4
LR_DECAY_EVERY = 50000
LR_DECAY_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-5
EPSILON = 1e-8
BETA1 = 0.9  # Adam: decay rate for the gradient average (momentum)
BETA2 = 0.999  # Adam: decay rate for the squared-gradient average
RANDOM_SEED = 42
NUM_MERGES = 4000  # BPE: how many subword merges to learn (the vocab-size dial)
PRINT_EVERY = 100
SAMPLE_EVERY = 250
SAVE_EVERY = 5000
TEMPERATURE = 0.8
TOP_K = 20

SCRIPT_DIR = Path(__file__).parent

assert EMBEDDING_DIM % NUM_HEADS == 0, "EMBEDDING_DIM must divide evenly into NUM_HEADS"

INV_SQRT_HEAD_DIM = DTYPE(HEAD_DIM**-0.5)
_CAUSAL_MASKS = {}
_ARANGES = {}


def causal_mask(seq_len):
    mask = _CAUSAL_MASKS.get(seq_len)
    if mask is None:
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        _CAUSAL_MASKS[seq_len] = mask
    return mask


def arange_cached(seq_len):
    values = _ARANGES.get(seq_len)
    if values is None:
        values = np.arange(seq_len)
        _ARANGES[seq_len] = values
    return values


# ----------------------------------------------------------------------------------
# Structure helpers — params/grads/memory all share the same nested shape
# (a dict, with "layers" being a list of per-layer dicts), so one recursive walk
# handles them uniformly.
# ----------------------------------------------------------------------------------


def zeros_like_structure(structure):
    """A zero-filled clone of any nested dict/list/ndarray structure."""
    if isinstance(structure, dict):
        return {key: zeros_like_structure(value) for key, value in structure.items()}
    if isinstance(structure, list):
        return [zeros_like_structure(value) for value in structure]
    return np.zeros_like(structure)


def to_serializable(structure):
    """Convert a nested structure of ndarrays into JSON-dumpable lists."""
    if isinstance(structure, dict):
        return {key: to_serializable(value) for key, value in structure.items()}
    if isinstance(structure, list):
        return [to_serializable(value) for value in structure]
    return structure.tolist()


def from_serializable(template, data):
    if isinstance(template, dict):
        return {key: from_serializable(template[key], data[key]) for key in template}
    if isinstance(template, list):
        return [from_serializable(item_template, item_data) for item_template, item_data in zip(template, data)]
    return np.asarray(data, dtype=template.dtype)


def infer_last_logged_iteration(loss_path):
    if not loss_path.exists():
        return 0
    last = None
    with open(loss_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if line.strip():
                last = line.split(",", 1)[0]
    return int(last) + 1 if last is not None else 0


def model_config(vocab_size):
    return {
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "ff_hidden_dim": FF_HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "sequence_length": SEQUENCE_LENGTH,
        "dtype": DTYPE.__name__,
    }


def assert_checkpoint_compatible(checkpoint, vocab_size):
    expected = model_config(vocab_size)
    found = checkpoint.get("model_config")
    if found != expected:
        raise ValueError(f"Checkpoint model config does not match this script. Expected {expected}, found {found}")


def current_learning_rate(base_lr, iteration, warmup, cosine_steps, min_lr):
    """Linear warmup to base_lr, then a cosine decay down to min_lr over cosine_steps
    steps, then hold at min_lr. cosine_steps <= 0 keeps LR constant after warmup."""
    if warmup and iteration < warmup:
        return base_lr * (iteration + 1) / warmup
    if cosine_steps and cosine_steps > 0:
        progress = (iteration - warmup) / cosine_steps
        progress = min(max(progress, 0.0), 1.0)
        return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress)))
    return base_lr


def adagrad_update(params, grads, memory):
    """In-place Adagrad update across the whole nested parameter structure, with
    per-element gradient clipping (the recursion bottoms out on ndarrays)."""
    if isinstance(params, dict):
        for key in params:
            adagrad_update(params[key], grads[key], memory[key])
    elif isinstance(params, list):
        for param, grad, mem in zip(params, grads, memory):
            adagrad_update(param, grad, mem)
    else:
        np.clip(grads, -GRADIENT_CLIP, GRADIENT_CLIP, out=grads)
        memory += grads * grads
        params -= LEARNING_RATE * grads / np.sqrt(memory + EPSILON)


def adam_update(params, grads, first_moment, second_moment, step, learning_rate):
    """In-place Adam update across the whole nested parameter structure.

    Bias correction is folded into one scalar step size so every ndarray update
    avoids allocating corrected moment arrays.
    """
    step_size = learning_rate * np.sqrt(1.0 - BETA2**step) / (1.0 - BETA1**step)
    _adam_update(params, grads, first_moment, second_moment, DTYPE(step_size))


def _adam_update(params, grads, first_moment, second_moment, step_size):
    if isinstance(params, dict):
        for key in params:
            _adam_update(params[key], grads[key], first_moment[key], second_moment[key], step_size)
    elif isinstance(params, list):
        for param, grad, m, v in zip(params, grads, first_moment, second_moment):
            _adam_update(param, grad, m, v, step_size)
    else:
        np.clip(grads, -GRADIENT_CLIP, GRADIENT_CLIP, out=grads)
        first_moment *= BETA1
        first_moment += (1.0 - BETA1) * grads
        second_moment *= BETA2
        second_moment += (1.0 - BETA2) * grads * grads
        params -= step_size * first_moment / (np.sqrt(second_moment) + EPSILON)


# ----------------------------------------------------------------------------------
# BPE tokeniser — learn subword merges from the corpus, then encode text to token ids.
# ----------------------------------------------------------------------------------


def get_pair_counts(word_freqs):
    """Count every adjacent symbol pair across all words, weighted by word frequency."""
    pair_counts = {}
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def merge_pair(pair, word_freqs):
    """Return a new word_freqs with every adjacent occurrence of `pair` fused into one symbol."""
    merged = pair[0] + pair[1]
    new_word_freqs = {}
    for word_tuple, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_word_freqs[tuple(new_word)] = new_word_freqs.get(tuple(new_word), 0) + freq
    return new_word_freqs


def train_bpe(corpus_text, num_merges):
    """Learn `num_merges` BPE merges. Returns (merges, final_word_freqs)."""
    word_freqs = {}
    for word in corpus_text.split():
        key = tuple(word)
        word_freqs[key] = word_freqs.get(key, 0) + 1

    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        word_freqs = merge_pair(best_pair, word_freqs)
    return merges, word_freqs


def encode_word(word, merges):
    """Apply the learned merges (in order) to one word, returning its tuple of subword tokens."""
    temp = {tuple(word): 1}
    for pair in merges:
        temp = merge_pair(pair, temp)
    return next(iter(temp))


def build_params(vocab_size):
    """Builds every parameter the model needs, small random weights throughout."""

    def layer():
        return {
            # Combined Q/K/V projections (all heads at once): (EMBEDDING_DIM, EMBEDDING_DIM).
            "Wq": (np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02).astype(DTYPE),
            "Wk": (np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02).astype(DTYPE),
            "Wv": (np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02).astype(DTYPE),
            # Output projection mixing the concatenated heads.
            "Wo": (np.random.randn(EMBEDDING_DIM, EMBEDDING_DIM) * 0.02).astype(DTYPE),
            # Add & Norm after attention.
            "ln1_gamma": np.ones(EMBEDDING_DIM, dtype=DTYPE),
            "ln1_beta": np.zeros(EMBEDDING_DIM, dtype=DTYPE),
            # Feed-forward: up to FF_HIDDEN_DIM and back down.
            "W1": (np.random.randn(FF_HIDDEN_DIM, EMBEDDING_DIM) * np.sqrt(2.0 / EMBEDDING_DIM)).astype(DTYPE),
            "b1": np.zeros(FF_HIDDEN_DIM, dtype=DTYPE),
            "W2": (np.random.randn(EMBEDDING_DIM, FF_HIDDEN_DIM) * np.sqrt(2.0 / FF_HIDDEN_DIM)).astype(DTYPE),
            "b2": np.zeros(EMBEDDING_DIM, dtype=DTYPE),
            # Add & Norm after feed-forward.
            "ln2_gamma": np.ones(EMBEDDING_DIM, dtype=DTYPE),
            "ln2_beta": np.zeros(EMBEDDING_DIM, dtype=DTYPE),
        }

    return {
        # Weight tying: token_embeddings doubles as the output projection (both are
        # (vocab_size, EMBEDDING_DIM)), so there is no separate output weight matrix.
        "token_embeddings": (np.random.randn(vocab_size, EMBEDDING_DIM) * 0.02).astype(DTYPE),
        "position_embeddings": (np.random.randn(SEQUENCE_LENGTH, EMBEDDING_DIM) * 0.02).astype(DTYPE),
        "layers": [layer() for _ in range(NUM_LAYERS)],
        "output_bias": np.zeros(vocab_size, dtype=DTYPE),
    }


# ----------------------------------------------------------------------------------
# Layer norm (operates per row = per position)
# ----------------------------------------------------------------------------------


def layer_norm_forward(x, gamma, beta):
    """Normalise each row (last axis = features) to ~mean-0/variance-1, then scale and
    shift. Works for any leading dims, e.g. (T, E) or (B, T, E)."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = np.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    std = np.sqrt(variance + LAYER_NORM_EPS)
    normalised = (x - mean) / std
    return gamma * normalised + beta, (normalised, std, gamma)


def layer_norm_backward(d_out, cache):
    """Returns (d_x, d_gamma, d_beta) for a layer norm. gamma/beta gradients sum over
    every axis except the last (the feature axis), so this handles (T, E) and (B, T, E)."""
    normalised, std, gamma = cache
    reduce_axes = tuple(range(d_out.ndim - 1))
    d_gamma = (d_out * normalised).sum(axis=reduce_axes)
    d_beta = d_out.sum(axis=reduce_axes)
    d_normalised = d_out * gamma
    d_x = (1.0 / std) * (d_normalised - d_normalised.mean(axis=-1, keepdims=True) - normalised * (d_normalised * normalised).mean(axis=-1, keepdims=True))
    return d_x, d_gamma, d_beta


# ----------------------------------------------------------------------------------
# Forward pass — whole sequence at once, returns probabilities + caches for backward
# ----------------------------------------------------------------------------------


def forward(params, input_ids):
    """Batched forward over input_ids of shape (B, T). Returns probabilities (B, T, V),
    the per-layer caches, and the final belt (B, T, EMBEDDING_DIM)."""
    batch, seq_len = input_ids.shape
    mask = causal_mask(seq_len)  # (T, T), True above the diagonal

    # Embedding + positional encoding -> the belt, shape (B, T, EMBEDDING_DIM)
    belt = params["token_embeddings"][input_ids] + params["position_embeddings"][:seq_len]

    caches = []
    for layer in params["layers"]:
        cache = {"input": belt}

        # ----- Multi-head causal self-attention -----
        # Project, then split the width into heads: (B, NUM_HEADS, T, HEAD_DIM).
        queries = (belt @ layer["Wq"].T).reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        keys = (belt @ layer["Wk"].T).reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        values = (belt @ layer["Wv"].T).reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * INV_SQRT_HEAD_DIM  # (B, H, T, T)
        scores[:, :, mask] = -1.0e9  # can't look at the future
        scores -= scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # attention weights

        context = weights @ values  # (B, H, T, HEAD_DIM)
        concat = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, EMBEDDING_DIM)  # glue heads back
        attention_out = concat @ layer["Wo"].T

        cache.update(queries=queries, keys=keys, values=values, weights=weights, concat=concat)

        # ----- Add & Norm 1 -----
        residual1 = belt + attention_out
        ln1, ln1_cache = layer_norm_forward(residual1, layer["ln1_gamma"], layer["ln1_beta"])
        cache["ln1"] = ln1
        cache["ln1_cache"] = ln1_cache

        # ----- Feed-forward -----
        hidden_pre = ln1 @ layer["W1"].T + layer["b1"]
        hidden = np.maximum(0.0, hidden_pre)  # ReLU
        feed_forward_out = hidden @ layer["W2"].T + layer["b2"]
        cache["hidden_pre"] = hidden_pre
        cache["hidden"] = hidden

        # ----- Add & Norm 2 -----
        residual2 = ln1 + feed_forward_out
        ln2, ln2_cache = layer_norm_forward(residual2, layer["ln2_gamma"], layer["ln2_beta"])
        cache["ln2_cache"] = ln2_cache

        belt = ln2
        caches.append(cache)

    # ----- Output projection + softmax, per position (weights tied to token embeddings) -----
    logits = belt @ params["token_embeddings"].T + params["output_bias"]
    logits -= logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    return probabilities, caches, belt


# ----------------------------------------------------------------------------------
# Backward pass — mirrors the forward in reverse, returns a full gradient structure
# ----------------------------------------------------------------------------------


def backward(params, input_ids, targets, probabilities, caches, final_belt):
    """Batched backward. input_ids/targets are (B, T); probabilities is (B, T, V).
    Weight gradients sum over both the batch and time axes (every prediction contributes)."""
    batch, seq_len = input_ids.shape
    grads = zeros_like_structure(params)
    mask = causal_mask(seq_len)
    batch_idx = np.arange(batch)[:, None]
    time_idx = np.arange(seq_len)[None, :]

    # ----- Output projection (softmax + cross-entropy gradient) -----
    d_logits = probabilities.copy()
    d_logits[batch_idx, time_idx, targets] -= 1.0
    d_logits /= DTYPE(batch * seq_len)  # mean over every predicted character in the batch
    flat_belt = final_belt.reshape(-1, EMBEDDING_DIM)
    # Tied output weights: this gradient lands on token_embeddings; the embedding-lookup
    # gradient is scatter-added onto it at the very end of the backward pass.
    grads["token_embeddings"] = d_logits.reshape(-1, d_logits.shape[-1]).T @ flat_belt
    grads["output_bias"] = d_logits.sum(axis=(0, 1))
    d_belt = d_logits @ params["token_embeddings"]

    # ----- Back through each block, last to first -----
    for layer_index in reversed(range(NUM_LAYERS)):
        layer = params["layers"][layer_index]
        cache = caches[layer_index]
        layer_grads = grads["layers"][layer_index]

        # Add & Norm 2
        d_residual2, layer_grads["ln2_gamma"], layer_grads["ln2_beta"] = layer_norm_backward(d_belt, cache["ln2_cache"])
        # residual2 = ln1 + feed_forward_out -> gradient flows to both
        d_feed_forward_out = d_residual2
        d_ln1 = d_residual2.copy()

        # Feed-forward backward (collapse B and T to one axis for the weight matmuls)
        layer_grads["W2"] = d_feed_forward_out.reshape(-1, EMBEDDING_DIM).T @ cache["hidden"].reshape(-1, FF_HIDDEN_DIM)
        layer_grads["b2"] = d_feed_forward_out.sum(axis=(0, 1))
        d_hidden = d_feed_forward_out @ layer["W2"]
        d_hidden_pre = d_hidden * (cache["hidden_pre"] > 0)  # ReLU gate
        layer_grads["W1"] = d_hidden_pre.reshape(-1, FF_HIDDEN_DIM).T @ cache["ln1"].reshape(-1, EMBEDDING_DIM)
        layer_grads["b1"] = d_hidden_pre.sum(axis=(0, 1))
        d_ln1 += d_hidden_pre @ layer["W1"]

        # Add & Norm 1
        d_residual1, layer_grads["ln1_gamma"], layer_grads["ln1_beta"] = layer_norm_backward(d_ln1, cache["ln1_cache"])
        # residual1 = belt_in + attention_out -> gradient flows to both
        d_attention_out = d_residual1
        d_belt_in = d_residual1.copy()

        # Attention backward
        layer_grads["Wo"] = d_attention_out.reshape(-1, EMBEDDING_DIM).T @ cache["concat"].reshape(-1, EMBEDDING_DIM)
        d_concat = d_attention_out @ layer["Wo"]
        d_context = d_concat.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # (B, H, T, HEAD_DIM)

        weights = cache["weights"]
        queries, keys, values = cache["queries"], cache["keys"], cache["values"]

        # Through the value-blend
        d_weights = d_context @ values.transpose(0, 1, 3, 2)
        d_values = weights.transpose(0, 1, 3, 2) @ d_context

        # Through the softmax
        d_scores = weights * (d_weights - (d_weights * weights).sum(axis=-1, keepdims=True))
        d_scores[:, :, mask] = 0.0
        d_scores *= INV_SQRT_HEAD_DIM

        # Through the scores into Q and K
        d_queries = d_scores @ keys
        d_keys = d_scores.transpose(0, 1, 3, 2) @ queries

        # Flatten heads back to (B, T, EMBEDDING_DIM)
        d_queries_flat = d_queries.transpose(0, 2, 1, 3).reshape(batch, seq_len, EMBEDDING_DIM)
        d_keys_flat = d_keys.transpose(0, 2, 1, 3).reshape(batch, seq_len, EMBEDDING_DIM)
        d_values_flat = d_values.transpose(0, 2, 1, 3).reshape(batch, seq_len, EMBEDDING_DIM)

        belt_in = cache["input"]
        flat_belt_in = belt_in.reshape(-1, EMBEDDING_DIM)
        layer_grads["Wq"] = d_queries_flat.reshape(-1, EMBEDDING_DIM).T @ flat_belt_in
        layer_grads["Wk"] = d_keys_flat.reshape(-1, EMBEDDING_DIM).T @ flat_belt_in
        layer_grads["Wv"] = d_values_flat.reshape(-1, EMBEDDING_DIM).T @ flat_belt_in
        d_belt_in += d_queries_flat @ layer["Wq"] + d_keys_flat @ layer["Wk"] + d_values_flat @ layer["Wv"]

        d_belt = d_belt_in  # gradient flowing to the block below

    # ----- Embeddings backward -----
    # d_belt is now the gradient on the initial (token + position) belt.
    np.add.at(grads["token_embeddings"], input_ids, d_belt)  # scatter-add (a char may repeat)
    grads["position_embeddings"][:seq_len] += d_belt.sum(axis=0)  # same positions across the batch

    return grads


def parse_args():
    parser = argparse.ArgumentParser(description="Train a NumPy BPE-subword transformer.")
    parser.add_argument("--max-iters", type=int, default=None, help="Stop after this many iterations; omit for an open-ended training run.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY, help="Print and log loss every N iterations.")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY, help="Generate a sample every N iterations; 0 disables sampling.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save checkpoint/weights every N iterations; 0 disables periodic saves.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Checkpoint path; defaults to output-dir/checkpoint.pkl.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint.pkl file.")
    parser.add_argument("--resume-weights", type=Path, default=None, help="Load legacy weights.json and restart Adam moments.")
    parser.add_argument("--start-iteration", type=int, default=None, help="Global iteration to use with --resume-weights; defaults to last loss.csv iteration + 1.")
    parser.add_argument("--reset-logs", action="store_true", help="Overwrite loss/sample logs even when resuming.")
    parser.add_argument("--lr", type=float, default=6e-4, help="Base Adam learning rate.")
    parser.add_argument("--cosine-steps", type=int, default=30000, help="Cosine-decay LR from base to min over this many steps (after warmup); 0 keeps it constant.")
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE, help="Floor the cosine schedule decays to.")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Tokens to generate per sample.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature; lower is safer, higher is wilder.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Only sample from the top K tokens; 0 uses the full distribution.")
    parser.add_argument("--sequential", action="store_true", help="Walk through the corpus in order instead of random windows.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs_shakespeare", help="Directory for loss, samples, and weights.")
    parser.add_argument("--corpus", type=Path, default=SCRIPT_DIR / "Original Data" / "Shakespeare.txt", help="Path to the training corpus text file.")
    parser.add_argument("--batch-size", type=int, default=16, help="Windows processed together per Adam step (effective batch size). 1 = original behaviour.")
    parser.add_argument("--warmup", type=int, default=1000, help="Linear LR warmup over this many optimizer steps; 0 disables.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of the corpus (its tail) held out for validation; 0 disables.")
    parser.add_argument("--val-batches", type=int, default=64, help="Number of fixed windows used to estimate validation loss.")
    return parser.parse_args()


def sample_next_id(probabilities, rng, temperature=TEMPERATURE, top_k=TOP_K):
    probs = probabilities.astype(np.float64, copy=True)
    temperature = max(float(temperature), 1.0e-6)
    if temperature != 1.0:
        logits = np.log(probs + EPSILON) / temperature
        logits -= logits.max()
        probs = np.exp(logits)

    if top_k and 0 < top_k < len(probs):
        keep = np.argpartition(probs, -top_k)[-top_k:]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered

    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        probs = np.full_like(probs, 1.0 / len(probs))
    else:
        probs /= total
    return int(rng.choice(len(probs), p=probs))


def main():
    """Reads the corpus, builds the model, trains it, samples periodically."""

    args = parse_args()
    if args.resume and args.resume_weights:
        raise ValueError("Use either --resume or --resume-weights, not both.")
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------- Corpus + vocabulary ----------
    corpus_path = args.corpus
    with open(corpus_path, encoding="utf-8") as f:
        corpus_text = f.read()

    # BPE tokenisation. Training the merges + encoding the corpus is slow in pure Python,
    # so we cache the result (keyed on corpus contents + NUM_MERGES) and reload it next time.
    corpus_hash = hashlib.md5(corpus_text.encode("utf-8")).hexdigest()
    cache_path = corpus_path.with_name(f"{corpus_path.stem}.bpe{NUM_MERGES}.pkl")

    cached = None
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            candidate = pickle.load(f)
        if candidate.get("hash") == corpus_hash and candidate.get("num_merges") == NUM_MERGES and candidate.get("scheme") == 2:
            cached = candidate

    if cached is not None:
        vocabulary = cached["vocabulary"]
        corpus_ids = cached["corpus_ids"]
        print(f"Loaded cached BPE from {cache_path.name} | {len(vocabulary)} tokens | {len(corpus_ids):,} ids")
    else:
        print(f"Training BPE ({NUM_MERGES} merges)... (one-time; result will be cached)")
        bpe_start = time.perf_counter()
        merges, merged_word_freqs = train_bpe(corpus_text, NUM_MERGES)

        # Vocabulary = every surviving subword + all base chars (fallback) + the word boundary.
        vocab_tokens = set()
        for word_tuple in merged_word_freqs:
            vocab_tokens.update(word_tuple)
        vocab_tokens.update(set(corpus_text))
        vocab_tokens.add("</w>")
        vocabulary = sorted(vocab_tokens)
        build_token_to_id = {tok: i for i, tok in enumerate(vocabulary)}
        boundary_id = build_token_to_id["</w>"]
        newline_id = build_token_to_id["\n"]

        # Encode the corpus to a flat id stream. Split into words AND newlines (re.findall
        # keeps each "\n" as its own unit): a word becomes its subword ids + a boundary token,
        # a newline becomes the newline token — so line breaks survive into the model.
        encode_memo = {}
        corpus_id_list = []
        for unit in re.findall(r"\S+|\n", corpus_text):
            if unit == "\n":
                corpus_id_list.append(newline_id)
                continue
            if unit not in encode_memo:
                encode_memo[unit] = [build_token_to_id[tok] for tok in encode_word(unit, merges)]
            corpus_id_list.extend(encode_memo[unit])
            corpus_id_list.append(boundary_id)

        corpus_ids = np.array(corpus_id_list, dtype=np.int64)
        with open(cache_path, "wb") as f:
            pickle.dump({"hash": corpus_hash, "num_merges": NUM_MERGES, "scheme": 2, "vocabulary": vocabulary, "corpus_ids": corpus_ids}, f)
        print(f"BPE done in {time.perf_counter() - bpe_start:.1f}s | {len(merges)} merges | {len(encode_memo):,} unique words | cached to {cache_path.name}")

    vocab_size = len(vocabulary)
    id_to_token = {i: tok for i, tok in enumerate(vocabulary)}
    corpus_length = len(corpus_ids)

    # ---------- Train / validation split (validation is the corpus tail) ----------
    if args.val_fraction and args.val_fraction > 0.0:
        split = int(corpus_length * (1.0 - args.val_fraction))
        train_ids = corpus_ids[:split]
        val_ids = corpus_ids[split:]
    else:
        train_ids = corpus_ids
        val_ids = None

    train_max_start = len(train_ids) - SEQUENCE_LENGTH - 1
    if train_max_start <= 0:
        raise ValueError(f"Training corpus must be longer than SEQUENCE_LENGTH={SEQUENCE_LENGTH}")

    val_starts = None
    if val_ids is not None:
        val_max_start = len(val_ids) - SEQUENCE_LENGTH - 1
        if val_max_start <= 0:
            val_ids = None  # held-out tail too short to evaluate
        else:
            val_starts = rng.integers(0, val_max_start + 1, size=args.val_batches)

    print(
        f"Corpus: {corpus_path.name} | length: {corpus_length:,} tokens | vocab size: {vocab_size} | "
        f"train/val: {len(train_ids):,}/{0 if val_ids is None else len(val_ids):,} | "
        f"layers: {NUM_LAYERS} | dim: {EMBEDDING_DIM} | heads: {NUM_HEADS} | ff: {FF_HIDDEN_DIM} | "
        f"ctx: {SEQUENCE_LENGTH} | batch: {args.batch_size} | dtype: {DTYPE.__name__}"
    )

    # ---------- Output directory ----------
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_path or (output_dir / "checkpoint.pkl")

    # ---------- Parameters + Adam moment estimates ----------
    params = build_params(vocab_size)
    first_moment = zeros_like_structure(params)
    second_moment = zeros_like_structure(params)
    running_loss = []
    running_loss_total = []
    pos = 0
    iteration = 0

    if args.resume:
        with open(args.resume, "rb") as f:
            checkpoint = pickle.load(f)
        assert_checkpoint_compatible(checkpoint, vocab_size)
        params = checkpoint["params"]
        first_moment = checkpoint["first_moment"]
        second_moment = checkpoint["second_moment"]
        running_loss = checkpoint.get("running_loss", [])[-1000:]
        running_loss_total = checkpoint.get("running_loss_total", [])[-1000:]
        pos = checkpoint.get("pos", 0)
        iteration = checkpoint.get("iteration", 0)
        rng.bit_generator.state = checkpoint["rng_state"]
        print(f"Resumed checkpoint {args.resume} at iteration {iteration:,}")
    elif args.resume_weights:
        with open(args.resume_weights, encoding="utf-8") as f:
            params = from_serializable(params, json.load(f))
        iteration = args.start_iteration if args.start_iteration is not None else infer_last_logged_iteration(output_dir / "loss.csv")
        print(f"Loaded legacy weights {args.resume_weights}; Adam moments restarted at iteration {iteration:,}")

    log_mode = "w" if args.reset_logs or (not args.resume and not args.resume_weights) or not (output_dir / "loss.csv").exists() else "a"
    loss_has_lr = True
    if log_mode == "w":
        (output_dir / "loss.csv").write_text("iteration,loss_total,loss_per_token,learning_rate\n", encoding="utf-8")
        (output_dir / "samples.txt").write_text("", encoding="utf-8")
        if val_ids is not None:
            (output_dir / "val_loss.csv").write_text("iteration,val_loss_per_token\n", encoding="utf-8")
    else:
        with open(output_dir / "loss.csv", encoding="utf-8") as f:
            loss_has_lr = "learning_rate" in f.readline().strip().split(",")

    config = model_config(vocab_size) | {
        "base_learning_rate": args.lr,
        "cosine_steps": args.cosine_steps,
        "min_lr": args.min_lr,
        "gradient_clip": GRADIENT_CLIP,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "val_fraction": args.val_fraction if val_ids is not None else 0.0,
        "corpus": str(corpus_path),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "random_seed": RANDOM_SEED,
        "checkpoint_path": str(checkpoint_path),
        "resume": str(args.resume) if args.resume else None,
        "resume_weights": str(args.resume_weights) if args.resume_weights else None,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    positions = arange_cached(SEQUENCE_LENGTH)

    # ---------- Sampling ----------
    def sample(seed_token_id, n_tokens):
        context = [seed_token_id]
        generated_tokens = []
        for _ in range(n_tokens):
            cropped = np.array(context[-SEQUENCE_LENGTH:], dtype=np.int64)[None, :]  # (1, t) batch
            probabilities, _, _ = forward(params, cropped)
            next_id = sample_next_id(probabilities[0, -1], rng, args.temperature, args.top_k)
            context.append(next_id)
            generated_tokens.append(id_to_token[next_id])
        # Detokenise: subwords concatenate; the boundary token becomes a space.
        return "".join(" " if tok == "</w>" else tok for tok in generated_tokens)

    # ---------- IO helpers ----------
    def save_weights():
        with open(output_dir / "weights.json", "w", encoding="utf-8") as f:
            json.dump(to_serializable(params), f)

    def save_checkpoint(next_iteration=None):
        checkpoint_iteration = iteration if next_iteration is None else next_iteration
        checkpoint = {
            "model_config": model_config(vocab_size),
            "params": params,
            "first_moment": first_moment,
            "second_moment": second_moment,
            "iteration": checkpoint_iteration,
            "pos": pos,
            "rng_state": rng.bit_generator.state,
            "running_loss": running_loss[-1000:],
            "running_loss_total": running_loss_total[-1000:],
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_weights()

    def append_loss(iteration, loss_total, loss, learning_rate):
        with open(output_dir / "loss.csv", "a", encoding="utf-8") as f:
            if loss_has_lr:
                f.write(f"{iteration},{loss_total},{loss},{learning_rate}\n")
            else:
                f.write(f"{iteration},{loss_total},{loss}\n")

    def append_sample(iteration, sample_text):
        with open(output_dir / "samples.txt", "a", encoding="utf-8") as f:
            f.write(f"=== Iteration {iteration} ===\n{sample_text}\n\n")

    def append_val(iteration, val_loss):
        with open(output_dir / "val_loss.csv", "a", encoding="utf-8") as f:
            f.write(f"{iteration},{val_loss}\n")

    def evaluate_val():
        """Mean loss/token over the fixed held-out windows, in one batched forward — no gradient."""
        inp = np.stack([val_ids[s : s + SEQUENCE_LENGTH] for s in val_starts])
        tgt = np.stack([val_ids[s + 1 : s + SEQUENCE_LENGTH + 1] for s in val_starts])
        probs, _, _ = forward(params, inp)
        bi = np.arange(inp.shape[0])[:, None]
        ti = positions[None, :]
        return float(np.mean(-np.log(probs[bi, ti, tgt] + EPSILON)))

    # ---------- Training loop ----------
    updates_this_run = 0
    last_print_update = 0
    training_start_time = time.perf_counter()
    last_print_time = training_start_time

    batch_size = max(1, args.batch_size)
    time_index = positions[None, :]
    batch_index = np.arange(batch_size)[:, None]

    while args.max_iters is None or updates_this_run < args.max_iters:
        # ----- Sample a batch of windows and process them together as one (B, T) tensor -----
        if args.sequential:
            starts = []
            for _ in range(batch_size):
                if pos > train_max_start:
                    pos = 0
                starts.append(pos)
                pos += SEQUENCE_LENGTH
            starts = np.array(starts)
        else:
            starts = rng.integers(0, train_max_start + 1, size=batch_size)

        inputs = np.stack([train_ids[s : s + SEQUENCE_LENGTH] for s in starts])
        targets = np.stack([train_ids[s + 1 : s + SEQUENCE_LENGTH + 1] for s in starts])

        # ----- Forward, loss, backward (the whole batch in one pass) -----
        probabilities, caches, final_belt = forward(params, inputs)
        token_losses = -np.log(probabilities[batch_index, time_index, targets] + EPSILON)  # (B, T)
        loss = float(token_losses.mean())
        loss_total = float(token_losses.sum() / batch_size)  # mean per-window total, comparable to batch-1
        running_loss_total.append(loss_total)
        running_loss.append(loss)

        grads = backward(params, inputs, targets, probabilities, caches, final_belt)

        # ----- Adam update (clips + updates every parameter in place) -----
        learning_rate = current_learning_rate(args.lr, iteration, args.warmup, args.cosine_steps, args.min_lr)
        adam_update(params, grads, first_moment, second_moment, iteration + 1, learning_rate)

        # ----- Periodic logging -----
        if args.print_every and iteration % args.print_every == 0:
            now = time.perf_counter()
            batch_elapsed = now - last_print_time
            total_elapsed = now - training_start_time
            completed = updates_this_run + 1 - last_print_update
            last_print_update = updates_this_run + 1
            last_print_time = now
            steps_per_sec = (completed / batch_elapsed) if batch_elapsed > 0 else 0.0
            recent_window = min(1000, len(running_loss))
            recent = running_loss[-recent_window:]
            recent_total = running_loss_total[-recent_window:]

            val_str = ""
            if val_ids is not None:
                val_loss = evaluate_val()
                append_val(iteration, val_loss)
                val_str = f" | val/token {val_loss:6.3f}"

            print(
                f"iter {iteration:>7d} | loss {loss_total:8.3f} | avg loss {mean(recent_total):8.3f} | "
                f"loss/token {loss:6.3f} | avg/token {mean(recent):6.3f}{val_str} | "
                f"lr {learning_rate:.2e} | {steps_per_sec:5.2f} step/s ({steps_per_sec * batch_size:6.1f} win/s) | "
                f"total {total_elapsed:7.2f}s"
            )
            append_loss(iteration, loss_total, loss, learning_rate)

        if args.sample_every and iteration % args.sample_every == 0:
            sample_text = sample(int(inputs[0, 0]), args.sample_size)
            recent_window = min(1000, len(running_loss))
            recent = running_loss[-recent_window:]
            recent_total = running_loss_total[-recent_window:]
            print(f"Sample at iteration {iteration}:\n\nAverage loss recent: {mean(recent_total):.4f}\nAverage loss/token recent: {mean(recent):.4f}\n\n--------\n{sample_text}\n--------\n")
            append_sample(iteration, sample_text)

        next_iteration = iteration + 1
        if args.save_every and next_iteration > 0 and next_iteration % args.save_every == 0:
            save_checkpoint(next_iteration)
            print(f"Saved checkpoint to {checkpoint_path}")

        iteration = next_iteration
        updates_this_run += 1

    save_checkpoint(iteration)
    elapsed = time.perf_counter() - training_start_time
    print(f"Finished {updates_this_run:,} updates in {elapsed:.2f}s; global iteration is {iteration:,}. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
