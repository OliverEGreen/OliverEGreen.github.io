"""
Trains an LSTM character-by-character. Aggressively optimised NumPy port of
raw_lstm.py — same algorithm, same hyperparameters, but with corpus
pre-encoding, pre-allocated activation buffers, vectorised loss, inlined
forward pass, and in-place updates throughout.

Mirror of basic_rnn_numpy.py's structure, expanded for LSTM's 14 parameters
and 4 gates per timestep.
"""

import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

# Constants
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

    # Pre-encode the corpus to int IDs once.
    corpus_ids = np.array([char_to_id[ch] for ch in corpus_text], dtype=np.int64)
    corpus_length = len(corpus_ids)

    # ---------- Output directory ----------
    output_dir = SCRIPT_DIR / "outputs_lstm_numpy"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "loss.csv").write_text("iteration,loss\n", encoding="utf-8")
    (output_dir / "samples.txt").write_text("", encoding="utf-8")

    # ---------- Parameters (14 total) ----------
    # Forget gate
    forget_gate_input_weights = np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01
    forget_gate_hidden_weights = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01
    forget_gate_bias = np.ones(HIDDEN_SIZE)  # Hochreiter's "remember everything" init

    # Input gate
    input_gate_input_weights = np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01
    input_gate_hidden_weights = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01
    input_gate_bias = np.zeros(HIDDEN_SIZE)

    # Candidate values
    candidate_gate_input_weights = np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01
    candidate_gate_hidden_weights = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01
    candidate_gate_bias = np.zeros(HIDDEN_SIZE)

    # Output gate
    output_gate_input_weights = np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01
    output_gate_hidden_weights = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01
    output_gate_bias = np.zeros(HIDDEN_SIZE)

    # Output projection
    hidden_to_output_weights = np.random.randn(vocab_size, HIDDEN_SIZE) * 0.01
    output_bias = np.zeros(vocab_size)

    # ---------- Adagrad memory accumulators (14, all zeros) ----------
    mem_forget_gate_input_weights = np.zeros_like(forget_gate_input_weights)
    mem_forget_gate_hidden_weights = np.zeros_like(forget_gate_hidden_weights)
    mem_forget_gate_bias = np.zeros_like(forget_gate_bias)

    mem_input_gate_input_weights = np.zeros_like(input_gate_input_weights)
    mem_input_gate_hidden_weights = np.zeros_like(input_gate_hidden_weights)
    mem_input_gate_bias = np.zeros_like(input_gate_bias)

    mem_candidate_gate_input_weights = np.zeros_like(candidate_gate_input_weights)
    mem_candidate_gate_hidden_weights = np.zeros_like(candidate_gate_hidden_weights)
    mem_candidate_gate_bias = np.zeros_like(candidate_gate_bias)

    mem_output_gate_input_weights = np.zeros_like(output_gate_input_weights)
    mem_output_gate_hidden_weights = np.zeros_like(output_gate_hidden_weights)
    mem_output_gate_bias = np.zeros_like(output_gate_bias)

    mem_hidden_to_output_weights = np.zeros_like(hidden_to_output_weights)
    mem_output_bias = np.zeros_like(output_bias)

    # ---------- Pre-allocated activation buffers ----------
    # Row 0 is the carried-in state; rows 1..T are this iteration's outputs.
    hidden_states = np.zeros((SEQUENCE_LENGTH + 1, HIDDEN_SIZE))
    cell_states = np.zeros((SEQUENCE_LENGTH + 1, HIDDEN_SIZE))

    # Gate values saved per timestep (needed for backward).
    forget_gates_buffer = np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))
    input_gates_buffer = np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))
    candidate_values_buffer = np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))
    output_gates_buffer = np.zeros((SEQUENCE_LENGTH, HIDDEN_SIZE))

    probabilities_buffer = np.zeros((SEQUENCE_LENGTH, vocab_size))

    # ---------- Pre-allocated gradient accumulators (reused, reset each iter) ----------
    d_forget_gate_input_weights = np.zeros_like(forget_gate_input_weights)
    d_forget_gate_hidden_weights = np.zeros_like(forget_gate_hidden_weights)
    d_forget_gate_bias = np.zeros_like(forget_gate_bias)

    d_input_gate_input_weights = np.zeros_like(input_gate_input_weights)
    d_input_gate_hidden_weights = np.zeros_like(input_gate_hidden_weights)
    d_input_gate_bias = np.zeros_like(input_gate_bias)

    d_candidate_gate_input_weights = np.zeros_like(candidate_gate_input_weights)
    d_candidate_gate_hidden_weights = np.zeros_like(candidate_gate_hidden_weights)
    d_candidate_gate_bias = np.zeros_like(candidate_gate_bias)

    d_output_gate_input_weights = np.zeros_like(output_gate_input_weights)
    d_output_gate_hidden_weights = np.zeros_like(output_gate_hidden_weights)
    d_output_gate_bias = np.zeros_like(output_gate_bias)

    d_hidden_to_output_weights = np.zeros_like(hidden_to_output_weights)
    d_output_bias = np.zeros_like(output_bias)

    d_next_hidden_state = np.zeros(HIDDEN_SIZE)
    d_next_cell_state = np.zeros(HIDDEN_SIZE)

    # Group (param, grad, memory) triples for the uniform update loop
    update_triples = [
        (forget_gate_input_weights, d_forget_gate_input_weights, mem_forget_gate_input_weights),
        (forget_gate_hidden_weights, d_forget_gate_hidden_weights, mem_forget_gate_hidden_weights),
        (forget_gate_bias, d_forget_gate_bias, mem_forget_gate_bias),
        (input_gate_input_weights, d_input_gate_input_weights, mem_input_gate_input_weights),
        (input_gate_hidden_weights, d_input_gate_hidden_weights, mem_input_gate_hidden_weights),
        (input_gate_bias, d_input_gate_bias, mem_input_gate_bias),
        (candidate_gate_input_weights, d_candidate_gate_input_weights, mem_candidate_gate_input_weights),
        (candidate_gate_hidden_weights, d_candidate_gate_hidden_weights, mem_candidate_gate_hidden_weights),
        (candidate_gate_bias, d_candidate_gate_bias, mem_candidate_gate_bias),
        (output_gate_input_weights, d_output_gate_input_weights, mem_output_gate_input_weights),
        (output_gate_hidden_weights, d_output_gate_hidden_weights, mem_output_gate_hidden_weights),
        (output_gate_bias, d_output_gate_bias, mem_output_gate_bias),
        (hidden_to_output_weights, d_hidden_to_output_weights, mem_hidden_to_output_weights),
        (output_bias, d_output_bias, mem_output_bias),
    ]

    grads_to_clip = [triple[1] for triple in update_triples]
    grads_to_zero = grads_to_clip  # same set: reset each iteration

    running_loss = []

    # ---------- Lightweight forward_step used only for sampling ----------
    def forward_step(char_id, previous_hidden_state, previous_cell_state):
        f = 1.0 / (1.0 + np.exp(-(forget_gate_input_weights[:, char_id] + forget_gate_hidden_weights @ previous_hidden_state + forget_gate_bias)))
        i = 1.0 / (1.0 + np.exp(-(input_gate_input_weights[:, char_id] + input_gate_hidden_weights @ previous_hidden_state + input_gate_bias)))
        g = np.tanh(candidate_gate_input_weights[:, char_id] + candidate_gate_hidden_weights @ previous_hidden_state + candidate_gate_bias)
        o = 1.0 / (1.0 + np.exp(-(output_gate_input_weights[:, char_id] + output_gate_hidden_weights @ previous_hidden_state + output_gate_bias)))

        new_cell_state = f * previous_cell_state + i * g
        new_hidden_state = o * np.tanh(new_cell_state)

        logits = hidden_to_output_weights @ new_hidden_state + output_bias
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return new_hidden_state, new_cell_state, exp_logits / exp_logits.sum()

    # ---------- Sampling ----------
    def sample(seed_char_id, n_chars, starting_hidden_state, starting_cell_state):
        h = starting_hidden_state.copy()
        c = starting_cell_state.copy()
        current_char_id = seed_char_id
        output_chars = []
        for _ in range(n_chars):
            h, c, probabilities = forward_step(current_char_id, h, c)
            current_char_id = int(np.random.choice(vocab_size, p=probabilities))
            output_chars.append(id_to_char[current_char_id])
        return "".join(output_chars)

    # ---------- IO helpers ----------
    def save_weights():
        with open(output_dir / "weights.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "forget_gate_input_weights": forget_gate_input_weights.tolist(),
                    "forget_gate_hidden_weights": forget_gate_hidden_weights.tolist(),
                    "forget_gate_bias": forget_gate_bias.tolist(),
                    "input_gate_input_weights": input_gate_input_weights.tolist(),
                    "input_gate_hidden_weights": input_gate_hidden_weights.tolist(),
                    "input_gate_bias": input_gate_bias.tolist(),
                    "candidate_gate_input_weights": candidate_gate_input_weights.tolist(),
                    "candidate_gate_hidden_weights": candidate_gate_hidden_weights.tolist(),
                    "candidate_gate_bias": candidate_gate_bias.tolist(),
                    "output_gate_input_weights": output_gate_input_weights.tolist(),
                    "output_gate_hidden_weights": output_gate_hidden_weights.tolist(),
                    "output_gate_bias": output_gate_bias.tolist(),
                    "hidden_to_output_weights": hidden_to_output_weights.tolist(),
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

    while True:
        # Fresh epoch when we'd walk off the end of the corpus
        if pos + SEQUENCE_LENGTH + 1 > corpus_length:
            pos = 0
            hidden_states[0].fill(0.0)
            cell_states[0].fill(0.0)

        # Direct slices into pre-encoded corpus
        inputs = corpus_ids[pos : pos + SEQUENCE_LENGTH]
        targets = corpus_ids[pos + 1 : pos + SEQUENCE_LENGTH + 1]

        # ----- Forward pass (inlined; writes directly into buffers) -----
        for t in range(SEQUENCE_LENGTH):
            h_prev = hidden_states[t]
            c_prev = cell_states[t]
            x_id = inputs[t]

            # Sigmoid is computed inline as 1 / (1 + exp(-x)) to avoid a helper call
            f = 1.0 / (1.0 + np.exp(-(forget_gate_input_weights[:, x_id] + forget_gate_hidden_weights @ h_prev + forget_gate_bias)))
            i = 1.0 / (1.0 + np.exp(-(input_gate_input_weights[:, x_id] + input_gate_hidden_weights @ h_prev + input_gate_bias)))
            g = np.tanh(candidate_gate_input_weights[:, x_id] + candidate_gate_hidden_weights @ h_prev + candidate_gate_bias)
            o = 1.0 / (1.0 + np.exp(-(output_gate_input_weights[:, x_id] + output_gate_hidden_weights @ h_prev + output_gate_bias)))

            forget_gates_buffer[t] = f
            input_gates_buffer[t] = i
            candidate_values_buffer[t] = g
            output_gates_buffer[t] = o

            cell_states[t + 1] = f * c_prev + i * g
            hidden_states[t + 1] = o * np.tanh(cell_states[t + 1])

            logits = hidden_to_output_weights @ hidden_states[t + 1] + output_bias
            logits -= logits.max()
            exp_logits = np.exp(logits)
            probabilities_buffer[t] = exp_logits / exp_logits.sum()

        # ----- Loss (vectorised) -----
        loss = float(-np.sum(np.log(probabilities_buffer[np.arange(SEQUENCE_LENGTH), targets] + EPSILON)))
        running_loss.append(loss)

        # ----- Backward pass (BPTT) -----
        # Reset gradient accumulators in place
        for grad in grads_to_zero:
            grad.fill(0.0)
        d_next_hidden_state.fill(0.0)
        d_next_cell_state.fill(0.0)

        for t in range(SEQUENCE_LENGTH - 1, -1, -1):
            # softmax + cross-entropy gradient
            dy = probabilities_buffer[t].copy()
            dy[targets[t]] -= 1.0

            # Output-projection gradients
            d_hidden_to_output_weights += np.outer(dy, hidden_states[t + 1])
            d_output_bias += dy

            # Backprop into hidden state at this timestep
            dh = hidden_to_output_weights.T @ dy + d_next_hidden_state

            # Split dh between output_gate path and cell-state-via-tanh path
            tanh_c = np.tanh(cell_states[t + 1])
            d_output_gate = dh * tanh_c
            dc = dh * output_gates_buffer[t] * (1.0 - tanh_c * tanh_c) + d_next_cell_state

            # Split dc four ways
            f_t = forget_gates_buffer[t]
            i_t = input_gates_buffer[t]
            g_t = candidate_values_buffer[t]
            o_t = output_gates_buffer[t]
            prev_c = cell_states[t]
            prev_h = hidden_states[t]

            d_forget_gate = dc * prev_c
            d_input_gate = dc * g_t
            d_candidate_values = dc * i_t
            d_next_cell_state = dc * f_t  # overwrite for the previous timestep

            # Backprop through each gate's activation
            d_forget_gate_pre = d_forget_gate * f_t * (1.0 - f_t)
            d_input_gate_pre = d_input_gate * i_t * (1.0 - i_t)
            d_candidate_pre = d_candidate_values * (1.0 - g_t * g_t)
            d_output_gate_pre = d_output_gate * o_t * (1.0 - o_t)

            # Accumulate gradients to each gate's 3 parameters.
            # The input-weights gradient uses the column-add trick (equivalent to outer product with a one-hot)
            d_forget_gate_input_weights[:, inputs[t]] += d_forget_gate_pre
            d_forget_gate_hidden_weights += np.outer(d_forget_gate_pre, prev_h)
            d_forget_gate_bias += d_forget_gate_pre

            d_input_gate_input_weights[:, inputs[t]] += d_input_gate_pre
            d_input_gate_hidden_weights += np.outer(d_input_gate_pre, prev_h)
            d_input_gate_bias += d_input_gate_pre

            d_candidate_gate_input_weights[:, inputs[t]] += d_candidate_pre
            d_candidate_gate_hidden_weights += np.outer(d_candidate_pre, prev_h)
            d_candidate_gate_bias += d_candidate_pre

            d_output_gate_input_weights[:, inputs[t]] += d_output_gate_pre
            d_output_gate_hidden_weights += np.outer(d_output_gate_pre, prev_h)
            d_output_gate_bias += d_output_gate_pre

            # Pass gradient back one timestep through the hidden state
            d_next_hidden_state = (
                forget_gate_hidden_weights.T @ d_forget_gate_pre
                + input_gate_hidden_weights.T @ d_input_gate_pre
                + candidate_gate_hidden_weights.T @ d_candidate_pre
                + output_gate_hidden_weights.T @ d_output_gate_pre
            )

        # ----- Gradient clipping (in-place) -----
        for grad in grads_to_clip:
            np.clip(grad, -GRADIENT_CLIP, GRADIENT_CLIP, out=grad)

        # ----- Adagrad parameter update (in-place across all 14 params) -----
        for param, grad, memory in update_triples:
            memory += grad * grad
            param -= LEARNING_RATE * grad / np.sqrt(memory + EPSILON)

        # ----- Persist final hidden + cell state of this sequence as next sequence's seed -----
        hidden_states[0] = hidden_states[SEQUENCE_LENGTH]
        cell_states[0] = cell_states[SEQUENCE_LENGTH]

        pos += SEQUENCE_LENGTH

        # ----- Periodic logging -----
        if iteration % 100 == 0:
            now = time.perf_counter()
            batch_elapsed = now - last_print_time
            total_elapsed = now - training_start_time
            last_print_time = now
            iters_per_sec = (100.0 / batch_elapsed) if batch_elapsed > 0 else 0.0
            overall_iters_per_sec = (iteration / total_elapsed) if total_elapsed > 0 else 0.0
            print(
                f"iter {iteration:>7d} | loss {loss:7.3f} | batch {batch_elapsed * 1000:6.1f}ms ({iters_per_sec:6.1f} it/s) | overall {overall_iters_per_sec:6.1f} it/s | total {total_elapsed:7.2f}s"
            )
            append_loss(iteration, loss)

        if iteration % 1000 == 0:
            sample_text = sample(inputs[0], SAMPLE_SIZE, hidden_states[0], cell_states[0])
            recent = running_loss[-1000:]
            print(f"Sample at iteration {iteration}:\n\nAverage loss last 1000: {mean(recent):.4f}\n\n--------\n{sample_text}\n--------\n")
            append_sample(iteration, sample_text)
            save_weights()

        iteration += 1


if __name__ == "__main__":
    main()
