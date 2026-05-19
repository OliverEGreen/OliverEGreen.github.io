"""
Trains an RNN character-by-character. Aggressively optimised NumPy port of
basic_rnn.py — same algorithm, same hyperparameters, but with corpus
pre-encoding, pre-allocated activation buffers, vectorised loss, inlined
forward pass, and in-place updates throughout.
"""

import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

# Constants — identical to basic_rnn.py
HIDDEN_SIZE = 512
SEQUENCE_LENGTH = 25
GRADIENT_CLIP = 5.0
LEARNING_RATE = 1e-1
EPSILON = 5e-2
SAMPLE_SIZE = 400
RANDOM_SEED = 42

SCRIPT_DIR = Path(__file__).parent


def main():
    """Reads the corpus, builds the model, trains it, samples periodically."""

    np.random.seed(RANDOM_SEED)

    # ---------- Corpus + vocabulary ----------
    corpus_path = SCRIPT_DIR / "Original Data" / "Shakespeare.txt"
    with open(corpus_path, encoding="utf-8") as f:
        corpus_text = f.read()

    chars = sorted(set(corpus_text))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    # Pre-encode the whole corpus to int IDs once. Slicing this array is
    # essentially free, so the training loop never has to do a per-iteration
    # list comprehension over characters.
    corpus_ids = np.array([char_to_id[ch] for ch in corpus_text], dtype=np.int64)
    corpus_length = len(corpus_ids)

    # ---------- Output directory ----------
    output_dir = SCRIPT_DIR / "outputs_numpy"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "loss.csv").write_text("iteration,loss\n", encoding="utf-8")
    (output_dir / "samples.txt").write_text("", encoding="utf-8")

    # ---------- Parameters ----------
    input_to_hidden_weights = np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01
    hidden_to_hidden_weights = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01
    hidden_to_output_weights = np.random.randn(vocab_size, HIDDEN_SIZE) * 0.01
    hidden_bias = np.zeros(HIDDEN_SIZE)
    output_bias = np.zeros(vocab_size)

    # ---------- Adagrad memory accumulators ----------
    mem_input_to_hidden_weights = np.zeros_like(input_to_hidden_weights)
    mem_hidden_to_hidden_weights = np.zeros_like(hidden_to_hidden_weights)
    mem_hidden_to_output_weights = np.zeros_like(hidden_to_output_weights)
    mem_hidden_bias = np.zeros_like(hidden_bias)
    mem_output_bias = np.zeros_like(output_bias)

    # ---------- Pre-allocated activation buffers ----------
    # hidden_states[0] is the carried-in state; hidden_states[1..T] are the
    # outputs of the T timesteps. Reused every iteration, no realloc.
    hidden_states = np.zeros((SEQUENCE_LENGTH + 1, HIDDEN_SIZE))
    probabilities_buffer = np.zeros((SEQUENCE_LENGTH, vocab_size))

    # Pre-allocated gradient accumulators (reset each iteration but reused)
    d_input_to_hidden_weights = np.zeros_like(input_to_hidden_weights)
    d_hidden_to_hidden_weights = np.zeros_like(hidden_to_hidden_weights)
    d_hidden_to_output_weights = np.zeros_like(hidden_to_output_weights)
    d_hidden_bias = np.zeros_like(hidden_bias)
    d_output_bias = np.zeros_like(output_bias)
    d_next_hidden_state = np.zeros(HIDDEN_SIZE)

    # Group the (param, grad, memory) triples once so the update loop is uniform
    update_triples = [
        (input_to_hidden_weights, d_input_to_hidden_weights, mem_input_to_hidden_weights),
        (hidden_to_hidden_weights, d_hidden_to_hidden_weights, mem_hidden_to_hidden_weights),
        (hidden_to_output_weights, d_hidden_to_output_weights, mem_hidden_to_output_weights),
        (hidden_bias, d_hidden_bias, mem_hidden_bias),
        (output_bias, d_output_bias, mem_output_bias),
    ]

    # Triples that need gradient clipping (just the gradient array each time)
    grads_to_clip = [
        d_input_to_hidden_weights,
        d_hidden_to_hidden_weights,
        d_hidden_to_output_weights,
        d_hidden_bias,
        d_output_bias,
    ]

    running_loss = []

    # ---------- Lightweight forward_step used only for sampling ----------
    def forward_step(char_id, previous_hidden_state):
        new_hidden_state = np.tanh(input_to_hidden_weights[:, char_id] + hidden_to_hidden_weights @ previous_hidden_state + hidden_bias)
        logits = hidden_to_output_weights @ new_hidden_state + output_bias
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return new_hidden_state, exp_logits / exp_logits.sum()

    # ---------- Sampling ----------
    def sample(seed_char_id, n_chars, starting_hidden_state):
        hidden_state = starting_hidden_state.copy()
        current_char_id = seed_char_id
        output_chars = []
        for _ in range(n_chars):
            hidden_state, probabilities = forward_step(current_char_id, hidden_state)
            current_char_id = int(np.random.choice(vocab_size, p=probabilities))
            output_chars.append(id_to_char[current_char_id])
        return "".join(output_chars)

    # ---------- IO helpers ----------
    def save_weights():
        with open(output_dir / "weights.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_to_hidden_weights": input_to_hidden_weights.tolist(),
                    "hidden_to_hidden_weights": hidden_to_hidden_weights.tolist(),
                    "hidden_to_output_weights": hidden_to_output_weights.tolist(),
                    "hidden_bias": hidden_bias.tolist(),
                    "output_bias": output_bias.tolist(),
                },
                f,
            )

    def append_loss(iteration, loss):
        with open(output_dir / "loss.csv", "a", encoding="utf-8") as f:
            f.write(f"{iteration},{loss}\n")

    def append_sample(iteration, sample_text):
        with open(output_dir / "samples.txt", "a", encoding="utf-8") as f:
            f.write(f"=== Iteration {iteration} ===\n{sample_text}\n\n")

    # ---------- Training loop ----------
    pos = 0
    iteration = 0
    training_start_time = time.perf_counter()
    last_print_time = training_start_time

    # Use the buffer's row 0 as the persistent "current" hidden state across iterations.
    # When we start a new epoch we zero it out.

    while True:
        # Fresh epoch when we'd walk off the end of the corpus
        if pos + SEQUENCE_LENGTH + 1 > corpus_length:
            pos = 0
            hidden_states[0].fill(0.0)

        # Direct slices into the pre-encoded corpus — no per-iteration list comp
        inputs = corpus_ids[pos : pos + SEQUENCE_LENGTH]
        targets = corpus_ids[pos + 1 : pos + SEQUENCE_LENGTH + 1]

        # ----- Forward pass (inlined; writes directly into buffers) -----
        for t in range(SEQUENCE_LENGTH):
            previous_hidden_state = hidden_states[t]
            new_hidden_state = np.tanh(input_to_hidden_weights[:, inputs[t]] + hidden_to_hidden_weights @ previous_hidden_state + hidden_bias)
            hidden_states[t + 1] = new_hidden_state

            logits = hidden_to_output_weights @ new_hidden_state + output_bias
            logits -= logits.max()
            exp_logits = np.exp(logits)
            probabilities_buffer[t] = exp_logits / exp_logits.sum()

        # ----- Loss (vectorised gather of the right probabilities) -----
        loss = float(-np.sum(np.log(probabilities_buffer[np.arange(SEQUENCE_LENGTH), targets] + EPSILON)))
        running_loss.append(loss)

        # ----- Backward pass (BPTT) -----
        # Reset gradient accumulators in place
        d_input_to_hidden_weights.fill(0.0)
        d_hidden_to_hidden_weights.fill(0.0)
        d_hidden_to_output_weights.fill(0.0)
        d_hidden_bias.fill(0.0)
        d_output_bias.fill(0.0)
        d_next_hidden_state.fill(0.0)

        for t in range(SEQUENCE_LENGTH - 1, -1, -1):
            # softmax + cross-entropy gradient: predicted - one_hot(target)
            dy = probabilities_buffer[t].copy()
            dy[targets[t]] -= 1.0

            # Output-layer gradients
            d_hidden_to_output_weights += np.outer(dy, hidden_states[t + 1])
            d_output_bias += dy

            # Backprop into hidden state (this step's output + future's gradient)
            dh = hidden_to_output_weights.T @ dy + d_next_hidden_state

            # Backprop through tanh
            dh_raw = (1.0 - hidden_states[t + 1] ** 2) * dh

            # Hidden-layer gradients
            d_hidden_bias += dh_raw
            d_input_to_hidden_weights[:, inputs[t]] += dh_raw
            d_hidden_to_hidden_weights += np.outer(dh_raw, hidden_states[t])

            # Carry gradient back one timestep
            d_next_hidden_state = hidden_to_hidden_weights.T @ dh_raw

        # ----- Gradient clipping (in-place) -----
        for grad in grads_to_clip:
            np.clip(grad, -GRADIENT_CLIP, GRADIENT_CLIP, out=grad)

        # ----- Adagrad parameter update (in-place across all 5 params) -----
        for param, grad, memory in update_triples:
            memory += grad * grad
            param -= LEARNING_RATE * grad / np.sqrt(memory + EPSILON)

        # ----- Persist final hidden state of this sequence as next sequence's seed -----
        # The forward pass already wrote it to hidden_states[SEQUENCE_LENGTH];
        # copy it into row 0 for the next iteration.
        hidden_states[0] = hidden_states[SEQUENCE_LENGTH]

        pos += SEQUENCE_LENGTH

        # ----- Periodic logging -----
        if iteration % 100 == 0:
            now = time.perf_counter()
            batch_elapsed = now - last_print_time
            total_elapsed = now - training_start_time
            last_print_time = now
            # Avoid div-by-zero on the very first iteration where batch_elapsed ~ 0
            iters_per_sec = (100.0 / batch_elapsed) if batch_elapsed > 0 else 0.0
            overall_iters_per_sec = (iteration / total_elapsed) if total_elapsed > 0 else 0.0
            print(
                f"iter {iteration:>7d} | loss {loss:7.3f} | batch {batch_elapsed * 1000:6.1f}ms ({iters_per_sec:6.1f} it/s) | overall {overall_iters_per_sec:6.1f} it/s | total {total_elapsed:7.2f}s"
            )
            append_loss(iteration, loss)

        if iteration % 1000 == 0:
            sample_text = sample(inputs[0], SAMPLE_SIZE, hidden_states[0])
            recent = running_loss[-1000:]
            print(f"Sample at iteration {iteration}:\n\nAverage loss last 1000: {mean(recent):.4f}\n\n--------\n{sample_text}\n--------\n")
            append_sample(iteration, sample_text)
            save_weights()

        iteration += 1


if __name__ == "__main__":
    main()
