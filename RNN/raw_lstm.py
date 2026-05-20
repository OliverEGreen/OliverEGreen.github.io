"""
Trains an LSTM character-by-character.
"""

import json
import math
import random
from pathlib import Path
from statistics import mean

# Constants
HIDDEN_SIZE = 100  # Determines the size of our ongoing 'memory' vector
SEQUENCE_LENGTH = 25
GRADIENT_CLIP = 5  # Stops us exploding towards infinity or negative infinity
LEARNING_RATE = 1e-1  # Using numbers from Adagrad as per Karpathy
EPSILON = 1e-8  # Another Karpathy numbers for using Adagrad approach
SAMPLE_SIZE = 400  # Number of chars to generate

SCRIPT_DIR = Path(__file__).parent


def instantiate_matrix(rows: int, cols: int):
    """Creates a randomly-initialised matrix of arbitrary size"""
    output_matrix = []
    for _ in range(rows):
        # Using smaller random range otherwise hidden activations will saturate immediately
        output_matrix.append([random.gauss(0, 1) * 0.01 for _ in range(cols)])
    return output_matrix


def one_hot(index: int, size: int):
    """Returns a sparse vector where given index is 1 and rest are zeros."""
    return [1.0 if i == index else 0.0 for i in range(size)]


def matrix_vector_multiply(matrix, vector):
    """Multiplies a matrix by a vector"""
    output_list = []
    for row in matrix:
        total = 0.0
        for r, v in zip(row, vector):
            total += r * v
        output_list.append(total)
    return output_list


def add_vectors(vector_1, vector_2):
    """Adds together two vectors of equal length"""
    if len(vector_1) != len(vector_2):
        return "Vectors are of unequal length!"
    return [i + j for i, j in zip(vector_1, vector_2)]


def sigmoid_vector(vector):
    """Returns the floating point number ran through the sigmoid function."""
    return [1 / (1 + math.exp(-x)) for x in vector]


def desigmoidify_vector(vector):
    """De-squishes a sigmoidified squished vector"""
    return [x * (1 - x) for x in vector]


def tanh_vector(vector):
    """Tanh acts as a control / hyperbolic function"""
    return [math.tanh(x) for x in vector]


def detanhify_vector(vector):
    """De-squishes a tanhified squished vector"""
    return [1 - x**2 for x in vector]


def softmax_vector(vector):
    """Another classic control mechanism"""
    max_v = max(vector)
    exp_list = [math.exp(v - max_v) for v in vector]
    exp_sum = sum(exp_list)
    return [e / exp_sum for e in exp_list]


def main():
    """The main function, reading and training our RNN model"""

    # Bringing in corpus data
    corpus_directory = SCRIPT_DIR / "Original Data" / "Shakespeare.txt"

    # All corpus text read raw to allow for punctuation, upper/lower case etc.
    with open(corpus_directory, encoding="utf-8") as corpus:
        corpus_text = corpus.read()

    # Output directory for saved weights, samples, and loss log. Starts fresh each run.
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "loss.csv").write_text("iteration,loss\n", encoding="utf-8")
    (output_dir / "samples.txt").write_text("", encoding="utf-8")

    # Reducing to just unique set of all characters
    chars = list(set(corpus_text))
    vocab_size = len(chars)

    # Establishing the two-way dicts, as is standard NLP practice.
    # Because unwrapping objects billions of times causes a huge performance hit.
    # Better to stick with primitives.
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    # Setting up matrices. This gets exhaustive in an LSTM as we need 4 gates per matrix.

    # Forget gates
    forget_gate_input_weights = instantiate_matrix(HIDDEN_SIZE, vocab_size)
    forget_gate_hidden_weights = instantiate_matrix(HIDDEN_SIZE, HIDDEN_SIZE)
    forget_gate_bias = [1.0] * HIDDEN_SIZE  # This starts with all 1s unlike our other starting points

    # Input gates
    input_gate_input_weights = instantiate_matrix(HIDDEN_SIZE, vocab_size)
    input_gate_hidden_weights = instantiate_matrix(HIDDEN_SIZE, HIDDEN_SIZE)
    input_gate_bias = [0.0] * HIDDEN_SIZE

    # Candidate gates
    candidate_gate_input_weights = instantiate_matrix(HIDDEN_SIZE, vocab_size)
    candidate_gate_hidden_weights = instantiate_matrix(HIDDEN_SIZE, HIDDEN_SIZE)
    candidate_gate_bias = [0.0] * HIDDEN_SIZE

    # Output gates
    output_gate_input_weights = instantiate_matrix(HIDDEN_SIZE, vocab_size)
    output_gate_hidden_weights = instantiate_matrix(HIDDEN_SIZE, HIDDEN_SIZE)
    output_gate_bias = [0.0] * HIDDEN_SIZE

    # This bridges our 'memory' to the output
    hidden_to_output_weights = instantiate_matrix(vocab_size, HIDDEN_SIZE)

    # Another vector of all zeros, where our next-char prediction is emitted.
    output_bias = [0.0] * vocab_size

    # Setting up our memory accumulators for Adagrad approach

    # Forget gates
    mem_forget_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
    mem_forget_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
    mem_forget_gate_bias = [0.0] * HIDDEN_SIZE  # This starts with all 1s unlike our other starting points

    # Input gates
    mem_input_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
    mem_input_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
    mem_input_gate_bias = [0.0] * HIDDEN_SIZE

    # Candidate gates
    mem_candidate_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
    mem_candidate_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
    mem_candidate_gate_bias = [0.0] * HIDDEN_SIZE

    # Output gates
    mem_output_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
    mem_output_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
    mem_output_gate_bias = [0.0] * HIDDEN_SIZE

    mem_hidden_to_output_weights = [[0.0] * HIDDEN_SIZE for _ in range(vocab_size)]
    mem_output_bias = [0.0] * vocab_size

    running_loss = []

    def forward_step(char_id: int, previous_hidden_state, previous_cell_state):
        """A forward step in our training. The new parameter initial cell state refers to the
        cell state which flows alongside the hidden state through time."""

        # Taking the one-hot vector to train our inputs
        input_one_hot = one_hot(char_id, vocab_size)

        # Forget gate vector
        forget_gate_input_contribution = matrix_vector_multiply(forget_gate_input_weights, input_one_hot)
        forget_gate_hidden_contribution = matrix_vector_multiply(forget_gate_hidden_weights, previous_hidden_state)
        forget_gate_pre_activation = add_vectors(add_vectors(forget_gate_input_contribution, forget_gate_hidden_contribution), forget_gate_bias)
        forget_gate = sigmoid_vector(forget_gate_pre_activation)

        # Input gates
        input_gate_input_contribution = matrix_vector_multiply(input_gate_input_weights, input_one_hot)
        input_gate_hidden_contribution = matrix_vector_multiply(input_gate_hidden_weights, previous_hidden_state)
        input_gate_pre_activation = add_vectors(add_vectors(input_gate_input_contribution, input_gate_hidden_contribution), input_gate_bias)
        input_gate = sigmoid_vector(input_gate_pre_activation)

        # Candidate gates
        candidate_gate_input_contribution = matrix_vector_multiply(candidate_gate_input_weights, input_one_hot)
        candidate_gate_hidden_contribution = matrix_vector_multiply(candidate_gate_hidden_weights, previous_hidden_state)
        candidate_gate_pre_activation = add_vectors(add_vectors(candidate_gate_input_contribution, candidate_gate_hidden_contribution), candidate_gate_bias)
        candidate_values = tanh_vector(candidate_gate_pre_activation)

        # Output gates
        output_gate_input_contribution = matrix_vector_multiply(output_gate_input_weights, input_one_hot)
        output_gate_hidden_contribution = matrix_vector_multiply(output_gate_hidden_weights, previous_hidden_state)
        output_gate_pre_activation = add_vectors(add_vectors(output_gate_input_contribution, output_gate_hidden_contribution), output_gate_bias)
        output_gate = sigmoid_vector(output_gate_pre_activation)

        # New cell state
        new_cell_state = add_vectors(multiply_vectors_elementwise(forget_gate, previous_cell_state), multiply_vectors_elementwise(input_gate, candidate_values))

        # New hidden state
        new_hidden_state = multiply_vectors_elementwise(tanh_vector(new_cell_state), output_gate)

        # Control function
        logits = add_vectors(matrix_vector_multiply(hidden_to_output_weights, new_hidden_state), output_bias)

        # Getting outputs - vector of probabilities of all characters being next
        probabilities = softmax_vector(logits)

        return new_hidden_state, new_cell_state, probabilities, forget_gate, input_gate, candidate_values, output_gate

    def backward_pass(
        inputs, targets, all_hidden_states, all_cell_states, all_forget_gates, all_input_gates, all_candidate_values, all_output_gates, all_probabilities, initial_hidden_state, initial_cell_state
    ):
        """Takes information from the forward_step and returns fourteen gradient structures.
        This is BPTT (backprop through time) and it's where things get a bit gnarly."""

        # Instantitating our gradient accumulators

        # Forget gates
        d_forget_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
        d_forget_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
        d_forget_gate_bias = [0.0] * HIDDEN_SIZE  # This starts with all 1s unlike our other starting points

        # Input gates
        d_input_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
        d_input_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
        d_input_gate_bias = [0.0] * HIDDEN_SIZE

        # Candidate gates
        d_candidate_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
        d_candidate_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
        d_candidate_gate_bias = [0.0] * HIDDEN_SIZE

        # Output gates
        d_output_gate_input_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
        d_output_gate_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
        d_output_gate_bias = [0.0] * HIDDEN_SIZE

        d_hidden_to_output_weights = [[0.0] * HIDDEN_SIZE for _ in range(vocab_size)]
        d_output_bias = [0.0] * vocab_size

        d_next_hidden_state = [0.0] * HIDDEN_SIZE
        d_next_cell_state = [0.0] * HIDDEN_SIZE

        # Looping backwards in time through our sequence of predictions
        for t in range(SEQUENCE_LENGTH - 1, -1, -1):
            # Effectively calculating all the deltas to tell us where to take our SGD step
            dy = subtract_vectors(all_probabilities[t], one_hot(targets[t], vocab_size))
            # Exhaustively calculating this timestep's contribution to our ongoing d_hidden_to_output matrix
            d_hidden_to_output_weights = add_matrices(d_hidden_to_output_weights, outer_product(dy, all_hidden_states[t]))
            d_output_bias = add_vectors(d_output_bias, dy)  # Add our loss gradient to the outputs for this timestep
            dh = add_vectors(matrix_vector_multiply(transpose_matrix(hidden_to_output_weights), dy), d_next_hidden_state)

            tanh_of_cell_states = tanh_vector(all_cell_states[t])

            # Computing gradient flowing into the output gate
            d_output_gate = multiply_vectors_elementwise(dh, tanh_of_cell_states)

            # Gradient flowing from hidden state path into cell state
            dc_from_h = multiply_vectors_elementwise(multiply_vectors_elementwise(dh, all_output_gates[t]), [1 - x**2 for x in tanh_of_cell_states])

            # Gradient on the new cell state
            dc = add_vectors(dc_from_h, d_next_cell_state)

            # Splitting dc four ways
            previous_cell_state = initial_cell_state if t == 0 else all_cell_states[t - 1]

            # This is where the real LSTM magic happens, folks! Like hell if I understand what's going on here ¯\_(ツ)_/¯
            d_forget_gate = multiply_vectors_elementwise(dc, previous_cell_state)
            d_input_gate = multiply_vectors_elementwise(dc, all_candidate_values[t])
            d_candidate_values = multiply_vectors_elementwise(dc, all_input_gates[t])
            d_next_cell_state = multiply_vectors_elementwise(dc, all_forget_gates[t])

            # Now we need to de-squishify our clever tanh/sigmoid functions to get pre-activation values back
            d_forget_gate_pre_activation = multiply_vectors_elementwise(d_forget_gate, desigmoidify_vector(all_forget_gates[t]))
            d_input_gate_pre_activation = multiply_vectors_elementwise(d_input_gate, desigmoidify_vector(all_input_gates[t]))
            d_candidate_values_pre_activation = multiply_vectors_elementwise(d_candidate_values, detanhify_vector(all_candidate_values[t]))
            d_output_gate_pre_activation = multiply_vectors_elementwise(d_output_gate, desigmoidify_vector(all_output_gates[t]))

            # It's gradient accumulatin' time
            previous_hidden_state = initial_hidden_state if t == 0 else all_hidden_states[t - 1]

            # Yet more of this nonsense
            d_forget_gate_input_weights = add_matrices(d_forget_gate_input_weights, outer_product(d_forget_gate_pre_activation, one_hot(inputs[t], vocab_size)))
            d_forget_gate_hidden_weights = add_matrices(d_forget_gate_hidden_weights, outer_product(d_forget_gate_pre_activation, previous_hidden_state))
            d_forget_gate_bias = add_vectors(d_forget_gate_bias, d_forget_gate_pre_activation)

            d_input_gate_input_weights = add_matrices(d_input_gate_input_weights, outer_product(d_input_gate_pre_activation, one_hot(inputs[t], vocab_size)))
            d_input_gate_hidden_weights = add_matrices(d_input_gate_hidden_weights, outer_product(d_input_gate_pre_activation, previous_hidden_state))
            d_input_gate_bias = add_vectors(d_input_gate_bias, d_input_gate_pre_activation)

            d_candidate_gate_input_weights = add_matrices(d_candidate_gate_input_weights, outer_product(d_candidate_values_pre_activation, one_hot(inputs[t], vocab_size)))
            d_candidate_gate_hidden_weights = add_matrices(d_candidate_gate_hidden_weights, outer_product(d_candidate_values_pre_activation, previous_hidden_state))
            d_candidate_gate_bias = add_vectors(d_candidate_gate_bias, d_candidate_values_pre_activation)

            d_output_gate_input_weights = add_matrices(d_output_gate_input_weights, outer_product(d_output_gate_pre_activation, one_hot(inputs[t], vocab_size)))
            d_output_gate_hidden_weights = add_matrices(d_output_gate_hidden_weights, outer_product(d_output_gate_pre_activation, previous_hidden_state))
            d_output_gate_bias = add_vectors(d_output_gate_bias, d_output_gate_pre_activation)

            # Here's the big nasty one, broken down into steps A-D for sanity purposes
            step_a = matrix_vector_multiply(transpose_matrix(forget_gate_hidden_weights), d_forget_gate_pre_activation)
            step_b = matrix_vector_multiply(transpose_matrix(input_gate_hidden_weights), d_input_gate_pre_activation)
            step_c = matrix_vector_multiply(transpose_matrix(candidate_gate_hidden_weights), d_candidate_values_pre_activation)
            step_d = matrix_vector_multiply(transpose_matrix(output_gate_hidden_weights), d_output_gate_pre_activation)

            d_next_hidden_state = add_vectors(add_vectors(add_vectors(step_a, step_b), step_c), step_d)

        # Returning all of the gradient structures
        return (
            d_forget_gate_input_weights,
            d_forget_gate_hidden_weights,
            d_forget_gate_bias,
            d_input_gate_input_weights,
            d_input_gate_hidden_weights,
            d_input_gate_bias,
            d_candidate_gate_input_weights,
            d_candidate_gate_hidden_weights,
            d_candidate_gate_bias,
            d_output_gate_input_weights,
            d_output_gate_hidden_weights,
            d_output_gate_bias,
            d_hidden_to_output_weights,
            d_output_bias,
        )

    def forward_sequence(char_ids, initial_hidden_state, initial_cell_state):
        """Takes a sequence of character IDs, and an initial hidden state, iterates through them all
        and tracks the probabilities of the next character"""
        current_hidden_state = initial_hidden_state
        current_cell_state = initial_cell_state

        running_hidden_states = []
        running_cell_states = []
        running_probabilities = []
        running_forget_gates = []
        running_input_gates = []
        running_candidate_values = []
        running_output_gates = []

        for c in char_ids:
            new_hidden_state, new_cell_state, probabilities, new_forget_gate, new_input_gate, new_candidate_value, new_output_gate = forward_step(c, current_hidden_state, current_cell_state)

            # Updating our running lists with the output of the forward step function above.
            running_hidden_states.append(new_hidden_state)
            running_cell_states.append(new_cell_state)
            running_probabilities.append(probabilities)
            running_forget_gates.append(new_forget_gate)
            running_input_gates.append(new_input_gate)
            running_candidate_values.append(new_candidate_value)
            running_output_gates.append(new_output_gate)

            # This assignment moves us forward a step in our loop
            current_hidden_state = new_hidden_state
            current_cell_state = new_cell_state

        return (
            running_hidden_states,
            running_cell_states,
            running_forget_gates,
            running_input_gates,
            running_candidate_values,
            running_output_gates,
            running_probabilities,
            current_hidden_state,
            current_cell_state,
        )

    def cross_entropy_loss(all_probabilities, targets):
        """Calculates the probability the model assigned the correct next character and takes
        the negative log of that. Rewards correct choice, punishes a confident incorrect choice."""
        return sum(-math.log(probability[target]) for probability, target in zip(all_probabilities, targets))

    def subtract_vectors(vec_1, vec_2):
        """Takes one vector away from the other. Must be of equal size"""
        if len(vec_1) != len(vec_2):
            return "Vectors must be of equal length"
        return [v_1 - v_2 for v_1, v_2 in zip(vec_1, vec_2)]

    def multiply_vectors_elementwise(vec_1, vec_2):
        """Multiplies two vectors together of equal length in sequence"""
        if len(vec_1) != len(vec_2):
            return "Vectors must be of equal length"
        return [v_1 * v_2 for v_1, v_2 in zip(vec_1, vec_2)]

    def outer_product(vec_1, vec_2):
        """Takes two vectors and multiplies every value by every other value"""
        output = []
        for val_1 in vec_1:
            products = []
            for val_2 in vec_2:
                products.append(val_1 * val_2)
            output.append(products)
        return output

    def transpose_matrix(matrix):
        """Takes a matrix of length X and depth Y and returns one of length Y and depth X"""
        no_rows = len(matrix)
        no_columns = len(matrix[0])

        output = []
        for i in range(no_columns):
            new_row = []
            for j in range(no_rows):
                new_row.append(matrix[j][i])
            output.append(new_row)
        return output

    def add_matrices(matrix_1, matrix_2):
        """Takes two matrices of the same shape and adds values element-wise"""
        return [add_vectors(row_1, row_2) for row_1, row_2 in zip(matrix_1, matrix_2)]

    def clamp_vector(vec):
        """Used to clamp gradients within set bounds"""
        return [max(-GRADIENT_CLIP, min(val, GRADIENT_CLIP)) for val in vec]

    def clamp_matrix(matrix):
        """Clamps the values of a matrix within set bounds"""
        return [clamp_vector(vec) for vec in matrix]

    def adagrad_update_vector(param, grad, memory):
        new_memory = [m + g**2 for m, g in zip(memory, grad)]
        new_param = [p - (LEARNING_RATE * g) / math.sqrt(m + EPSILON) for m, p, g in zip(new_memory, param, grad)]
        return (new_param, new_memory)

    def adagrad_update_matrix(param, grad, memory):
        row_results = [adagrad_update_vector(p, g, m) for p, g, m in zip(param, grad, memory)]
        new_param, new_memory = zip(*row_results)
        return list(new_param), list(new_memory)

    def sample(seed_char_id, n_chars, starting_hidden_state, starting_cell_state):
        """Returns a string of generated text of length n_chars"""
        hidden_state = list(starting_hidden_state)  # Making a copy
        cell_state = list(starting_cell_state)
        current_char_id = seed_char_id
        output_chars = []

        for i in range(n_chars):
            new_hidden_state, new_cell_state, probabilities, _, _, _, _ = forward_step(current_char_id, hidden_state, cell_state)
            random_char_id = random.choices(range(vocab_size), weights=probabilities, k=1)[0]
            random_char = id_to_char[random_char_id]
            output_chars.append(random_char)
            current_char_id = random_char_id
            hidden_state = new_hidden_state
            cell_state = new_cell_state

        return "".join(output_chars)

    def save_weights():
        """Dumps the parameters to outputs/weights.json (overwrites each time)."""
        with open(output_dir / "weights.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "forget_gate_input_weights": forget_gate_input_weights,
                    "forget_gate_hidden_weights": forget_gate_hidden_weights,
                    "forget_gate_bias": forget_gate_bias,
                    "input_gate_input_weights": input_gate_input_weights,
                    "input_gate_hidden_weights": input_gate_hidden_weights,
                    "input_gate_bias": input_gate_bias,
                    "candidate_gate_input_weights": candidate_gate_input_weights,
                    "candidate_gate_hidden_weights": candidate_gate_hidden_weights,
                    "candidate_gate_bias": candidate_gate_bias,
                    "output_gate_input_weights": output_gate_input_weights,
                    "output_gate_hidden_weights": output_gate_hidden_weights,
                    "output_gate_bias": output_gate_bias,
                    "hidden_to_output_weights": hidden_to_output_weights,
                    "output_bias": output_bias,
                },
                f,
            )

    def append_loss(iteration, loss):
        """Appends one row to outputs/loss.csv."""
        with open(output_dir / "loss.csv", "a", encoding="utf-8") as f:
            f.write(f"{iteration},{loss}\n")

    def append_sample(iteration, sample_text):
        """Appends a labelled sample block to outputs/samples.txt."""
        with open(output_dir / "samples.txt", "a", encoding="utf-8") as f:
            f.write(f"=== Iteration {iteration} ===\n{sample_text}\n\n")

    pos = 0
    iteration = 0
    current_hidden_state = [0.0] * HIDDEN_SIZE
    current_cell_state = [0.0] * HIDDEN_SIZE

    # We begin our training loop
    while True:
        # Resetting for a fresh epoch once we've reached the end of our character sequence
        if pos + SEQUENCE_LENGTH + 1 > len(corpus_text):
            pos = 0  # Resets the position to zero, beginning a new loop
            current_hidden_state = [0.0] * HIDDEN_SIZE  # Resets current hidden state back to zeros
            current_cell_state = [0.0] * HIDDEN_SIZE

        # Grabbing chars, getting their IDs
        inputs = [char_to_id[x] for x in corpus_text[pos : pos + SEQUENCE_LENGTH]]
        targets = [char_to_id[x] for x in corpus_text[pos + 1 : pos + SEQUENCE_LENGTH + 1]]
        all_hidden_states, all_cell_states, all_forget_gates, all_input_gates, all_candidate_values, all_output_gates, all_probabilities, final_hidden_state, final_cell_state = forward_sequence(
            inputs, current_hidden_state, current_cell_state
        )
        loss = cross_entropy_loss(all_probabilities, targets)
        running_loss.append(loss)

        # Unpacking this monster tuple, telling us how much to nudge everything by (gradient)
        (
            d_forget_gate_input_weights,
            d_forget_gate_hidden_weights,
            d_forget_gate_bias,
            d_input_gate_input_weights,
            d_input_gate_hidden_weights,
            d_input_gate_bias,
            d_candidate_gate_input_weights,
            d_candidate_gate_hidden_weights,
            d_candidate_gate_bias,
            d_output_gate_input_weights,
            d_output_gate_hidden_weights,
            d_output_gate_bias,
            d_hidden_to_output_weights,
            d_output_bias,
        ) = backward_pass(
            inputs,
            targets,
            all_hidden_states,
            all_cell_states,
            all_forget_gates,
            all_input_gates,
            all_candidate_values,
            all_output_gates,
            all_probabilities,
            current_hidden_state,  # this becomes initial_hidden_state inside backward
            current_cell_state,  # this becomes initial_cell_state
        )

        # Clamping our 8+1 matrices
        c_forget_gate_input_weights = clamp_matrix(d_forget_gate_input_weights)
        c_forget_gate_hidden_weights = clamp_matrix(d_forget_gate_hidden_weights)

        c_input_gate_input_weights = clamp_matrix(d_input_gate_input_weights)
        c_input_gate_hidden_weights = clamp_matrix(d_input_gate_hidden_weights)

        c_candidate_gate_input_weights = clamp_matrix(d_candidate_gate_input_weights)
        c_candidate_gate_hidden_weights = clamp_matrix(d_candidate_gate_hidden_weights)

        c_output_gate_input_weights = clamp_matrix(d_output_gate_input_weights)
        c_output_gate_hidden_weights = clamp_matrix(d_output_gate_hidden_weights)

        c_hidden_to_output_weights = clamp_matrix(d_hidden_to_output_weights)

        # Clamping our 4+1 vectors
        c_forget_gate_bias = clamp_vector(d_forget_gate_bias)
        c_input_gate_bias = clamp_vector(d_input_gate_bias)
        c_candidate_gate_bias = clamp_vector(d_candidate_gate_bias)
        c_output_gate_bias = clamp_vector(d_output_gate_bias)

        c_output_bias = clamp_vector(d_output_bias)

        # Adagrad-updating all 14 parameters with their corresponding memory accumulators
        forget_gate_input_weights, mem_forget_gate_input_weights = adagrad_update_matrix(forget_gate_input_weights, c_forget_gate_input_weights, mem_forget_gate_input_weights)
        forget_gate_hidden_weights, mem_forget_gate_hidden_weights = adagrad_update_matrix(forget_gate_hidden_weights, c_forget_gate_hidden_weights, mem_forget_gate_hidden_weights)

        input_gate_input_weights, mem_input_gate_input_weights = adagrad_update_matrix(input_gate_input_weights, c_input_gate_input_weights, mem_input_gate_input_weights)
        input_gate_hidden_weights, mem_input_gate_hidden_weights = adagrad_update_matrix(input_gate_hidden_weights, c_input_gate_hidden_weights, mem_input_gate_hidden_weights)

        candidate_gate_input_weights, mem_candidate_gate_input_weights = adagrad_update_matrix(candidate_gate_input_weights, c_candidate_gate_input_weights, mem_candidate_gate_input_weights)
        candidate_gate_hidden_weights, mem_candidate_gate_hidden_weights = adagrad_update_matrix(candidate_gate_hidden_weights, c_candidate_gate_hidden_weights, mem_candidate_gate_hidden_weights)

        output_gate_input_weights, mem_output_gate_input_weights = adagrad_update_matrix(output_gate_input_weights, c_output_gate_input_weights, mem_output_gate_input_weights)
        output_gate_hidden_weights, mem_output_gate_hidden_weights = adagrad_update_matrix(output_gate_hidden_weights, c_output_gate_hidden_weights, mem_output_gate_hidden_weights)

        hidden_to_output_weights, mem_hidden_to_output_weights = adagrad_update_matrix(hidden_to_output_weights, c_hidden_to_output_weights, mem_hidden_to_output_weights)

        forget_gate_bias, mem_forget_gate_bias = adagrad_update_vector(forget_gate_bias, c_forget_gate_bias, mem_forget_gate_bias)
        input_gate_bias, mem_input_gate_bias = adagrad_update_vector(input_gate_bias, c_input_gate_bias, mem_input_gate_bias)
        candidate_gate_bias, mem_candidate_gate_bias = adagrad_update_vector(candidate_gate_bias, c_candidate_gate_bias, mem_candidate_gate_bias)
        output_gate_bias, mem_output_gate_bias = adagrad_update_vector(output_gate_bias, c_output_gate_bias, mem_output_gate_bias)

        output_bias, mem_output_bias = adagrad_update_vector(output_bias, c_output_bias, mem_output_bias)

        # Moving our pointer along
        pos += SEQUENCE_LENGTH

        # Recording our training loss (i.e. accuracy) as we iterate
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Loss: {loss}")
            append_loss(iteration, loss)
        if iteration % 1000 == 0:
            sample_text = sample(inputs[0], SAMPLE_SIZE, current_hidden_state, current_cell_state)
            print(f"Sample at iteration {iteration}:\n\nAverage loss last 1000: {mean(running_loss[-10:])}\n\n--------\n{sample_text}\n--------\n")
            append_sample(iteration, sample_text)
            save_weights()

        # All of the end-of-loop resets
        iteration += 1
        current_hidden_state = final_hidden_state
        current_cell_state = final_cell_state


if __name__ == "__main__":
    main()
