"""
Trains an RNN character-by-character.
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
SAMPLE_SIZE = 400 # Number of chars to generate

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


def tanh_vector(vector):
    """Tanh acts as a control / hyperbolic function"""
    return [math.tanh(x) for x in vector]


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

    # Setting up matrices

    # This bridges raw input to hidden.
    input_to_hidden_weights = instantiate_matrix(HIDDEN_SIZE, vocab_size)

    # This matrix handles the 'memory'.
    hidden_to_hidden_weights = instantiate_matrix(HIDDEN_SIZE, HIDDEN_SIZE)

    # This bridges our 'memory' to the output
    hidden_to_output_weights = instantiate_matrix(vocab_size, HIDDEN_SIZE)

    # A vector initiated with all zeros.
    hidden_bias = [0.0] * HIDDEN_SIZE

    # Another vector of all zeros, where our next-char prediction is emitted.
    output_bias = [0.0] * vocab_size

    # Setting up our memory accumulators for Adagrad approach

    mem_input_to_hidden_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
    mem_hidden_to_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
    mem_hidden_to_output_weights = [[0.0] * HIDDEN_SIZE for _ in range(vocab_size)]
    mem_hidden_bias = [0.0] * HIDDEN_SIZE
    mem_output_bias = [0.0] * vocab_size

    running_loss = []

    def forward_step(char_id: int, previous_hidden_state):

        # Taking the one-hot vector to train our inputs
        x = one_hot(char_id, vocab_size)

        # Ascertaining its influence on the hidden weights
        input_contribution = matrix_vector_multiply(input_to_hidden_weights, x)

        memory_contribution = matrix_vector_multiply(hidden_to_hidden_weights, previous_hidden_state)
        pre_activation = add_vectors(add_vectors(input_contribution, memory_contribution), hidden_bias)
        new_hidden_state = tanh_vector(pre_activation)

        # Control function
        logits = add_vectors(matrix_vector_multiply(hidden_to_output_weights, new_hidden_state), output_bias)

        # Getting outputs - vector of probabilities of all characters being next
        probabilities = softmax_vector(logits)

        return new_hidden_state, probabilities

    def backward_pass(inputs, targets, all_hidden_states, all_probabilities, initial_hidden_state):
        """Takes information from the forward_step and returns five gradient structures.
        This is BPTT (backprop through time) and it's where things get a bit gnarly."""

        # Instantitating our gradient accumulators
        d_input_to_hidden_weights = [[0.0] * vocab_size for _ in range(HIDDEN_SIZE)]
        d_hidden_to_hidden_weights = [[0.0] * HIDDEN_SIZE for _ in range(HIDDEN_SIZE)]
        d_hidden_to_output_weights = [[0.0] * HIDDEN_SIZE for _ in range(vocab_size)]
        d_hidden_bias = [0.0] * HIDDEN_SIZE
        d_output_bias = [0.0] * vocab_size

        d_next_hidden_state = [0.0] * HIDDEN_SIZE

        # Looping backwards in time through our sequence of predictions
        for t in range(SEQUENCE_LENGTH - 1, -1, -1):
            dy = subtract_vectors(all_probabilities[t], one_hot(targets[t], vocab_size))
            # Exhaustively calculating this timestep's contribution to our ongoing d_hidden_to_output matrix
            d_hidden_to_output_weights = add_matrices(d_hidden_to_output_weights, outer_product(dy, all_hidden_states[t]))
            d_output_bias = add_vectors(d_output_bias, dy)  # Add our loss gradient to the outputs for this timestep
            dh = add_vectors(matrix_vector_multiply(transpose_matrix(hidden_to_output_weights), dy), d_next_hidden_state)

            # Undoing the tanh squashification, lets us get the gradient we need
            dh_raw = multiply_vectors_elementwise(dh, [1 - x**2 for x in all_hidden_states[t]])

            # Updating matrices, doing the actual backprop steps
            d_hidden_bias = add_vectors(d_hidden_bias, dh_raw)
            d_input_to_hidden_weights = add_matrices(outer_product(dh_raw, one_hot(inputs[t], vocab_size)), d_input_to_hidden_weights)
            d_hidden_to_hidden_weights = add_matrices(d_hidden_to_hidden_weights, outer_product(dh_raw, initial_hidden_state if t == 0 else all_hidden_states[t - 1]))

            # Final step, updating the next hidden state matrix so we can continue our looping
            d_next_hidden_state = matrix_vector_multiply(transpose_matrix(hidden_to_hidden_weights), dh_raw)

        # Returning all fo the gradient structures
        return d_input_to_hidden_weights, d_hidden_to_hidden_weights, d_hidden_to_output_weights, d_hidden_bias, d_output_bias

    def forward_sequence(char_ids, initial_hidden_state):
        """Takes a sequence of character IDs, and an initial hidden state, iterates through them all
        and tracks the probabilities of the next character"""
        current_hidden_state = initial_hidden_state

        running_hidden_states = []
        running_probabilities = []

        for c in char_ids:
            new_hidden_state, probabilities = forward_step(c, current_hidden_state)
            running_hidden_states.append(new_hidden_state)
            running_probabilities.append(probabilities)

            # This assignment moves us forward a step in our loop
            current_hidden_state = new_hidden_state

        return (running_hidden_states, running_probabilities, current_hidden_state)

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

    def sample(seed_char_id, n_chars, starting_hidden_state):
        """Returns a string of generated text of length n_chars"""
        hidden_state = list(starting_hidden_state)  # Making a copy
        current_char_id = seed_char_id
        output_chars = []

        for i in range(n_chars):
            new_hidden_state, probabilities = forward_step(current_char_id, hidden_state)
            random_char_id = random.choices(range(vocab_size), weights=probabilities, k=1)[0]
            random_char = id_to_char[random_char_id]
            output_chars.append(random_char)
            current_char_id = random_char_id
            hidden_state = new_hidden_state

        return "".join(output_chars)

    def save_weights():
        """Dumps the five parameters to outputs/weights.json (overwrites each time)."""
        with open(output_dir / "weights.json", "w", encoding="utf-8") as f:
            json.dump({
                "input_to_hidden_weights": input_to_hidden_weights,
                "hidden_to_hidden_weights": hidden_to_hidden_weights,
                "hidden_to_output_weights": hidden_to_output_weights,
                "hidden_bias": hidden_bias,
                "output_bias": output_bias,
            }, f)

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

    # We begin our training loop
    while True:
        # Resetting for a fresh epoch once we've reached the end of our character sequence
        if pos + SEQUENCE_LENGTH + 1 > len(corpus_text):
            pos = 0  # Resets the position to zero, beginning a new loop
            current_hidden_state = [0.0] * HIDDEN_SIZE  # Resets current hidden state back to zeros

        # Grabbing chars, getting their IDs
        inputs = [char_to_id[x] for x in corpus_text[pos : pos + SEQUENCE_LENGTH]]
        targets = [char_to_id[x] for x in corpus_text[pos + 1 : pos + SEQUENCE_LENGTH + 1]]
        all_hidden_states, all_probabilities, final_hidden_state = forward_sequence(inputs, current_hidden_state)
        loss = cross_entropy_loss(all_probabilities, targets)
        running_loss.append(loss)

        # Unpacking this monster tuple, telling us how much to nudge everything by (gradient)
        d_input_to_hidden_weights, d_hidden_to_hidden_weights, d_hidden_to_output_weights, d_hidden_bias, d_output_bias = backward_pass(
            inputs, targets, all_hidden_states, all_probabilities, current_hidden_state
        )

        # Clamping gradient values to avoid gradient explosion
        c_input_to_hidden_weights = clamp_matrix(d_input_to_hidden_weights)
        c_hidden_to_hidden_weights = clamp_matrix(d_hidden_to_hidden_weights)
        c_hidden_to_output_weights = clamp_matrix(d_hidden_to_output_weights)
        c_hidden_bias = clamp_vector(d_hidden_bias)
        c_output_bias = clamp_vector(d_output_bias)

        # Updating all of our original matrices, memory-keeper matrices (from Adagrad approach) and vectors
        input_to_hidden_weights, mem_input_to_hidden_weights = adagrad_update_matrix(input_to_hidden_weights, c_input_to_hidden_weights, mem_input_to_hidden_weights)
        hidden_to_hidden_weights, mem_hidden_to_hidden_weights = adagrad_update_matrix(hidden_to_hidden_weights, c_hidden_to_hidden_weights, mem_hidden_to_hidden_weights)
        hidden_to_output_weights, mem_hidden_to_output_weights = adagrad_update_matrix(hidden_to_output_weights, c_hidden_to_output_weights, mem_hidden_to_output_weights)
        hidden_bias, mem_hidden_bias = adagrad_update_vector(hidden_bias, c_hidden_bias, mem_hidden_bias)
        output_bias, mem_output_bias = adagrad_update_vector(output_bias, c_output_bias, mem_output_bias)

        # Moving our pointer along
        pos += SEQUENCE_LENGTH

        # Recording our training loss (i.e. accuracy) as we iterate
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Loss: {loss}")
            append_loss(iteration, loss)
        if iteration % 1000 == 0:
            sample_text = sample(inputs[0], SAMPLE_SIZE, current_hidden_state)
            print(f"Sample at iteration {iteration}:\n\nAverage loss last 1000: {mean(running_loss[-10:])}\n\n--------\n{sample_text}\n--------\n")
            append_sample(iteration, sample_text)
            save_weights()

        iteration += 1
        current_hidden_state = final_hidden_state


if __name__ == "__main__":
    main()
