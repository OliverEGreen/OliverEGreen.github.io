"""
Trains a decoder-only (GPT-style) transformer character-by-character. Aggressively
optimised NumPy port of raw_transformers.py, tuned for Apple Silicon with float32
arrays, cached masks, random corpus windows, and low-allocation Adam updates.

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
import json
import pickle
import time
from pathlib import Path
from statistics import mean

import numpy as np

# M1/Accelerate is very fast at float32 matrix math; float64 roughly doubles memory
# bandwidth for no useful gain in this character model.
DTYPE = np.float32

# Quality/performance defaults tuned for Apple Silicon NumPy.
EMBEDDING_DIM = 128
NUM_HEADS = 8
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
FF_HIDDEN_DIM = 512
LAYER_NORM_EPS = 1e-5
NUM_LAYERS = 4
SEQUENCE_LENGTH = 128
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
PRINT_EVERY = 100
SAMPLE_EVERY = 1000
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


def add_scaled_into(accumulator, grads, scale):
    """accumulator += grads * scale, in place across the shared nested structure.
    Used to average per-window gradients into one batch gradient before the Adam step."""
    if isinstance(accumulator, dict):
        for key in accumulator:
            add_scaled_into(accumulator[key], grads[key], scale)
    elif isinstance(accumulator, list):
        for acc, grad in zip(accumulator, grads):
            add_scaled_into(acc, grad, scale)
    else:
        accumulator += grads * scale


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


def current_learning_rate(base_lr, iteration, decay_every, decay_factor, min_lr, warmup=0):
    if warmup and iteration < warmup:
        return base_lr * (iteration + 1) / warmup  # linear warmup from ~0 to base_lr
    if decay_every <= 0 or decay_factor >= 1.0:
        return base_lr
    decay_steps = iteration // decay_every
    return max(min_lr, base_lr * (decay_factor**decay_steps))


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
        "token_embeddings": (np.random.randn(vocab_size, EMBEDDING_DIM) * 0.02).astype(DTYPE),
        "position_embeddings": (np.random.randn(SEQUENCE_LENGTH, EMBEDDING_DIM) * 0.02).astype(DTYPE),
        "layers": [layer() for _ in range(NUM_LAYERS)],
        "hidden_to_output_weights": (np.random.randn(vocab_size, EMBEDDING_DIM) * 0.02).astype(DTYPE),
        "output_bias": np.zeros(vocab_size, dtype=DTYPE),
    }


# ----------------------------------------------------------------------------------
# Layer norm (operates per row = per position)
# ----------------------------------------------------------------------------------


def layer_norm_forward(x, gamma, beta):
    """Normalise each row to ~mean-0/variance-1, then scale and shift. Returns the
    output and a cache for the backward pass."""
    mean = x.mean(axis=1, keepdims=True)
    variance = np.mean((x - mean) * (x - mean), axis=1, keepdims=True)
    std = np.sqrt(variance + LAYER_NORM_EPS)
    normalised = (x - mean) / std
    return gamma * normalised + beta, (normalised, std, gamma)


def layer_norm_backward(d_out, cache):
    """Returns (d_x, d_gamma, d_beta) for a layer norm."""
    normalised, std, gamma = cache
    d_gamma = (d_out * normalised).sum(axis=0)
    d_beta = d_out.sum(axis=0)
    d_normalised = d_out * gamma
    d_x = (1.0 / std) * (d_normalised - d_normalised.mean(axis=1, keepdims=True) - normalised * (d_normalised * normalised).mean(axis=1, keepdims=True))
    return d_x, d_gamma, d_beta


# ----------------------------------------------------------------------------------
# Forward pass — whole sequence at once, returns probabilities + caches for backward
# ----------------------------------------------------------------------------------


def forward(params, input_ids):
    seq_len = len(input_ids)
    mask = causal_mask(seq_len)  # True above the diagonal

    # Embedding + positional encoding -> the belt, shape (T, EMBEDDING_DIM)
    belt = params["token_embeddings"][input_ids] + params["position_embeddings"][:seq_len]

    caches = []
    for layer in params["layers"]:
        cache = {"input": belt}

        # ----- Multi-head causal self-attention -----
        # Project, then split the width into (NUM_HEADS, T, HEAD_DIM).
        queries = (belt @ layer["Wq"].T).reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
        keys = (belt @ layer["Wk"].T).reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
        values = (belt @ layer["Wv"].T).reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)

        scores = (queries @ keys.transpose(0, 2, 1)) * INV_SQRT_HEAD_DIM  # (H, T, T)
        scores[:, mask] = -1.0e9  # can't look at the future
        scores -= scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # attention weights

        context = weights @ values  # (H, T, HEAD_DIM)
        concat = context.transpose(1, 0, 2).reshape(seq_len, EMBEDDING_DIM)  # glue heads back
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

    # ----- Output projection + softmax, per position -----
    logits = belt @ params["hidden_to_output_weights"].T + params["output_bias"]
    logits -= logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    return probabilities, caches, belt


# ----------------------------------------------------------------------------------
# Backward pass — mirrors the forward in reverse, returns a full gradient structure
# ----------------------------------------------------------------------------------


def backward(params, input_ids, targets, probabilities, caches, final_belt):
    seq_len = len(input_ids)
    grads = zeros_like_structure(params)
    mask = causal_mask(seq_len)
    positions = arange_cached(seq_len)

    # ----- Output projection (softmax + cross-entropy gradient) -----
    d_logits = probabilities.copy()
    d_logits[positions, targets] -= 1.0
    d_logits /= DTYPE(seq_len)
    grads["hidden_to_output_weights"] = d_logits.T @ final_belt
    grads["output_bias"] = d_logits.sum(axis=0)
    d_belt = d_logits @ params["hidden_to_output_weights"]

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

        # Feed-forward backward
        layer_grads["W2"] = d_feed_forward_out.T @ cache["hidden"]
        layer_grads["b2"] = d_feed_forward_out.sum(axis=0)
        d_hidden = d_feed_forward_out @ layer["W2"]
        d_hidden_pre = d_hidden * (cache["hidden_pre"] > 0)  # ReLU gate
        layer_grads["W1"] = d_hidden_pre.T @ cache["ln1"]
        layer_grads["b1"] = d_hidden_pre.sum(axis=0)
        d_ln1 += d_hidden_pre @ layer["W1"]

        # Add & Norm 1
        d_residual1, layer_grads["ln1_gamma"], layer_grads["ln1_beta"] = layer_norm_backward(d_ln1, cache["ln1_cache"])
        # residual1 = belt_in + attention_out -> gradient flows to both
        d_attention_out = d_residual1
        d_belt_in = d_residual1.copy()

        # Attention backward
        layer_grads["Wo"] = d_attention_out.T @ cache["concat"]
        d_concat = d_attention_out @ layer["Wo"]
        d_context = d_concat.reshape(seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)  # (H, T, HEAD_DIM)

        weights = cache["weights"]
        queries, keys, values = cache["queries"], cache["keys"], cache["values"]

        # Through the value-blend
        d_weights = d_context @ values.transpose(0, 2, 1)
        d_values = weights.transpose(0, 2, 1) @ d_context

        # Through the softmax
        d_scores = weights * (d_weights - (d_weights * weights).sum(axis=-1, keepdims=True))
        d_scores[:, mask] = 0.0
        d_scores *= INV_SQRT_HEAD_DIM

        # Through the scores into Q and K
        d_queries = d_scores @ keys
        d_keys = d_scores.transpose(0, 2, 1) @ queries

        # Flatten heads back to (T, EMBEDDING_DIM)
        d_queries_flat = d_queries.transpose(1, 0, 2).reshape(seq_len, EMBEDDING_DIM)
        d_keys_flat = d_keys.transpose(1, 0, 2).reshape(seq_len, EMBEDDING_DIM)
        d_values_flat = d_values.transpose(1, 0, 2).reshape(seq_len, EMBEDDING_DIM)

        belt_in = cache["input"]
        layer_grads["Wq"] = d_queries_flat.T @ belt_in
        layer_grads["Wk"] = d_keys_flat.T @ belt_in
        layer_grads["Wv"] = d_values_flat.T @ belt_in
        d_belt_in += d_queries_flat @ layer["Wq"] + d_keys_flat @ layer["Wk"] + d_values_flat @ layer["Wv"]

        d_belt = d_belt_in  # gradient flowing to the block below

    # ----- Embeddings backward -----
    # d_belt is now the gradient on the initial (token + position) belt.
    np.add.at(grads["token_embeddings"], input_ids, d_belt)  # scatter-add (a char may repeat)
    grads["position_embeddings"][:seq_len] += d_belt

    return grads


def parse_args():
    parser = argparse.ArgumentParser(description="Train a NumPy character transformer on Shakespeare.")
    parser.add_argument("--max-iters", type=int, default=None, help="Stop after this many iterations; omit for an open-ended training run.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY, help="Print and log loss every N iterations.")
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY, help="Generate a sample every N iterations; 0 disables sampling.")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save checkpoint/weights every N iterations; 0 disables periodic saves.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Checkpoint path; defaults to output-dir/checkpoint.pkl.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint.pkl file.")
    parser.add_argument("--resume-weights", type=Path, default=None, help="Load legacy weights.json and restart Adam moments.")
    parser.add_argument("--start-iteration", type=int, default=None, help="Global iteration to use with --resume-weights; defaults to last loss.csv iteration + 1.")
    parser.add_argument("--reset-logs", action="store_true", help="Overwrite loss/sample logs even when resuming.")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Base Adam learning rate.")
    parser.add_argument("--lr-decay-every", type=int, default=2500, help="Decay LR every N global iterations; 0 disables decay.")
    parser.add_argument("--lr-decay-factor", type=float, default=LR_DECAY_FACTOR, help="Multiplier applied at each LR decay step.")
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE, help="Floor for decayed LR.")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Characters to generate per sample.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature; lower is safer, higher is wilder.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Only sample from the top K chars; 0 uses the full distribution.")
    parser.add_argument("--sequential", action="store_true", help="Walk through the corpus in order instead of random windows.")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs_hp", help="Directory for loss, samples, and weights.")
    parser.add_argument("--corpus", type=Path, default=SCRIPT_DIR / "Original Data" / "HarryPotter.txt", help="Path to the training corpus text file.")
    parser.add_argument("--batch-size", type=int, default=16, help="Windows accumulated per Adam step (effective batch size). 1 = original behaviour.")
    parser.add_argument("--warmup", type=int, default=200, help="Linear LR warmup over this many optimizer steps; 0 disables.")
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

    chars = sorted(set(corpus_text))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    corpus_ids = np.array([char_to_id[ch] for ch in corpus_text], dtype=np.int64)
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
        f"Corpus: {corpus_path.name} | length: {corpus_length:,} chars | vocab size: {vocab_size} | "
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
        (output_dir / "loss.csv").write_text("iteration,loss_total,loss_per_char,learning_rate\n", encoding="utf-8")
        (output_dir / "samples.txt").write_text("", encoding="utf-8")
        if val_ids is not None:
            (output_dir / "val_loss.csv").write_text("iteration,val_loss_per_char\n", encoding="utf-8")
    else:
        with open(output_dir / "loss.csv", encoding="utf-8") as f:
            loss_has_lr = "learning_rate" in f.readline().strip().split(",")

    config = model_config(vocab_size) | {
        "base_learning_rate": args.lr,
        "lr_decay_every": args.lr_decay_every,
        "lr_decay_factor": args.lr_decay_factor,
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
    def sample(seed_char_id, n_chars):
        context = [seed_char_id]
        output_chars = []
        for _ in range(n_chars):
            cropped = np.array(context[-SEQUENCE_LENGTH:], dtype=np.int64)
            probabilities, _, _ = forward(params, cropped)
            next_id = sample_next_id(probabilities[-1], rng, args.temperature, args.top_k)
            context.append(next_id)
            output_chars.append(id_to_char[next_id])
        return "".join(output_chars)

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
        """Mean loss/char over the fixed held-out windows — no gradient, pure measurement."""
        losses = []
        for s in val_starts:
            s = int(s)
            inp = val_ids[s : s + SEQUENCE_LENGTH]
            tgt = val_ids[s + 1 : s + SEQUENCE_LENGTH + 1]
            probs, _, _ = forward(params, inp)
            losses.append(float(np.mean(-np.log(probs[positions, tgt] + EPSILON))))
        return mean(losses)

    # ---------- Training loop ----------
    updates_this_run = 0
    last_print_update = 0
    training_start_time = time.perf_counter()
    last_print_time = training_start_time

    batch_size = max(1, args.batch_size)

    while args.max_iters is None or updates_this_run < args.max_iters:
        # ----- Accumulate gradients over a batch of windows, then ONE Adam step -----
        accumulated = zeros_like_structure(params)
        batch_loss_total = 0.0
        batch_loss = 0.0
        last_inputs = None
        for _ in range(batch_size):
            if args.sequential:
                if pos > train_max_start:
                    pos = 0
                start = pos
                pos += SEQUENCE_LENGTH
            else:
                start = int(rng.integers(0, train_max_start + 1))

            inputs = train_ids[start : start + SEQUENCE_LENGTH]
            targets = train_ids[start + 1 : start + SEQUENCE_LENGTH + 1]
            last_inputs = inputs

            probabilities, caches, final_belt = forward(params, inputs)
            token_losses = -np.log(probabilities[positions, targets] + EPSILON)
            batch_loss_total += float(np.sum(token_losses))
            batch_loss += float(np.mean(token_losses))

            grads = backward(params, inputs, targets, probabilities, caches, final_belt)
            add_scaled_into(accumulated, grads, 1.0 / batch_size)  # average across the batch

        loss_total = batch_loss_total / batch_size
        loss = batch_loss / batch_size
        running_loss_total.append(loss_total)
        running_loss.append(loss)

        # ----- Adam update on the averaged gradient (clips + updates in place) -----
        learning_rate = current_learning_rate(args.lr, iteration, args.lr_decay_every, args.lr_decay_factor, args.min_lr, args.warmup)
        adam_update(params, accumulated, first_moment, second_moment, iteration + 1, learning_rate)

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

            val_str = ""
            if val_ids is not None:
                val_loss = evaluate_val()
                append_val(iteration, val_loss)
                val_str = f" | val/char {val_loss:6.3f}"

            print(
                f"iter {iteration:>7d} | loss/char {loss:6.3f} | avg/char {mean(recent):6.3f}{val_str} | "
                f"lr {learning_rate:.2e} | {steps_per_sec:5.2f} step/s ({steps_per_sec * batch_size:6.1f} win/s) | "
                f"total {total_elapsed:7.2f}s"
            )
            append_loss(iteration, loss_total, loss, learning_rate)

        if args.sample_every and iteration % args.sample_every == 0:
            sample_text = sample(int(last_inputs[0]), args.sample_size)
            recent_window = min(1000, len(running_loss))
            recent = running_loss[-recent_window:]
            print(f"Sample at iteration {iteration}:\n\nAverage loss/char recent: {mean(recent):.4f}\n\n--------\n{sample_text}\n--------\n")
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
