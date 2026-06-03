"""
Trains a language model to generate Shakespeare character-by-character using transformer architecture.
"""

import math
import random
from pathlib import Path

# Constants
EMBEDDING_DIM = 32
NUM_HEADS = 4
HEAD_DIM = 8
FF_HIDDEN_DIM = 128
LAYER_NORM_EPS = 1e-5
NUM_LAYERS = 2
SEQUENCE_LENGTH = 25
SAMPLE_SIZE = 200  # Number of chars to generate

GRADIENT_CLIP = 5  # Stops us exploding towards infinity or negative infinity
LEARNING_RATE = 1e-1  # Using numbers from Adagrad as per Karpathy
EPSILON = 1e-8  # Another Karpathy numbers for using Adagrad approach

SCRIPT_DIR = Path(__file__).parent


def instantiate_matrix(rows: int, cols: int):
    """Creates a randomly-initialised matrix of arbitrary size"""
    output_matrix = []
    for _ in range(rows):
        # Using smaller random range otherwise hidden activations will saturate immediately
        output_matrix.append([random.gauss(0, 1) * 0.01 for _ in range(cols)])
    return output_matrix


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


def softmax_vector(vector):
    """Another classic control mechanism"""
    max_v = max(vector)
    exp_list = [math.exp(v - max_v) for v in vector]
    exp_sum = sum(exp_list)
    return [e / exp_sum for e in exp_list]


def scale_vector(vector, scalar):
    """Scales a vector by a scalar"""
    return [v * scalar for v in vector]


def dot_product(vector_1, vector_2):
    """Returns the sum of two vectors when multiplied elementwise"""
    return sum(vec_1 * vec_2 for vec_1, vec_2 in zip(vector_1, vector_2))


def relu_vector(vector):
    """Applies the ReLU function to a vector, keeping positives and flattening negatives to zero"""
    return [max(0.0, v) for v in vector]


def main():
    """The main function, reading and training our language model"""

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

    # Reducing to unique set of all characters, sorted for consistency
    chars = sorted(set(corpus_text))
    vocab_size = len(chars)

    # Establishing the two-way dicts, as is standard NLP practice.
    # Because unwrapping objects billions of times causes a huge performance hit.
    # Better to stick with primitives.
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}

    # Setting up embedding tables
    token_embeddings = instantiate_matrix(vocab_size, EMBEDDING_DIM)
    position_embeddings = instantiate_matrix(SEQUENCE_LENGTH, EMBEDDING_DIM)

    layers = [
        {
            # Setting up the heart of our transformer
            "query_weights": [instantiate_matrix(HEAD_DIM, EMBEDDING_DIM) for _ in range(NUM_HEADS)],
            "key_weights": [instantiate_matrix(HEAD_DIM, EMBEDDING_DIM) for _ in range(NUM_HEADS)],
            "value_weights": [instantiate_matrix(HEAD_DIM, EMBEDDING_DIM) for _ in range(NUM_HEADS)],
            # After each attention run we reassemble our outputs together here
            "output_weights": instantiate_matrix(EMBEDDING_DIM, EMBEDDING_DIM),
            # These are the learnable scale and shift for the layer-norm that follows the attention sub-layer.
            "ln1_gamma": [1.0] * EMBEDDING_DIM,
            "ln1_beta": [0.0] * EMBEDDING_DIM,
            # Feed forward-bits. This is where the useful work gets done. It's also pleasingly symmetrical!
            "ff_up_weights": instantiate_matrix(FF_HIDDEN_DIM, EMBEDDING_DIM),
            "ff_up_bias": [0.0] * FF_HIDDEN_DIM,
            "ff_down_weights": instantiate_matrix(EMBEDDING_DIM, FF_HIDDEN_DIM),
            "ff_down_bias": [0.0] * EMBEDDING_DIM,
            "ln2_gamma": [1.0] * EMBEDDING_DIM,
            "ln2_beta": [0.0] * EMBEDDING_DIM,
        }
        for _ in range(NUM_LAYERS)
    ]

    hidden_to_output_weights = instantiate_matrix(vocab_size, EMBEDDING_DIM)
    output_bias = [0.0] * vocab_size

    def scaled_dot_product_attention(queries, keys, values):
        seq_len = len(queries)

        # Computing the scale factor as square root fo the head dimension
        scale_factor = 1 / (HEAD_DIM**0.5)

        outputs = []

        for i in range(seq_len):
            scores = []
            for j in range(i + 1):
                scores.append(dot_product(queries[i], keys[j]) * scale_factor)
            attention_weights = softmax_vector(scores)
            accumulator = [0.0] * HEAD_DIM
            for j in range(i + 1):
                # Blending in past information
                accumulator = add_vectors(accumulator, scale_vector(values[j], attention_weights[j]))
            outputs.append(accumulator)
        return outputs

    # The critical function, we'll be doing this A LOT.
    def multi_head_attention(belt, layer):
        head_outputs = []

        for h in range(NUM_HEADS):
            Wq, Wk, Wv = layer["query_weights"][h], layer["key_weights"][h], layer["value_weights"][h]
            queries = [matrix_vector_multiply(Wq, x) for x in belt]
            keys = [matrix_vector_multiply(Wk, x) for x in belt]
            values = [matrix_vector_multiply(Wv, x) for x in belt]
            head_outputs.append(scaled_dot_product_attention(queries, keys, values))

        # Concatenate the heads, per position
        concatenated_list = []
        for i in range(len(belt)):
            row = []
            for h in range(NUM_HEADS):
                row.extend(head_outputs[h][i])
            concatenated_list.append(row)

        return [matrix_vector_multiply(layer["output_weights"], c) for c in concatenated_list]

    def forward(input_ids):
        final_belt = []

        # Building out our starting belt
        for position, char_id in enumerate(input_ids):
            final_belt.append(add_vectors(token_embeddings[char_id], position_embeddings[position]))

        intermediate_dicts = []

        # Looping through all layers and updating our belt as we go along
        for layer in layers:
            final_belt, intermediate_dict = transformer_block(final_belt, layer)
            intermediate_dicts.append(intermediate_dict)

        # It's all matmul, forever. And adding in some bias ofc.
        probabilities = [add_vectors(matrix_vector_multiply(hidden_to_output_weights, x), output_bias) for x in final_belt]

        return [softmax_vector(p) for p in probabilities], final_belt, intermediate_dicts

    def feed_forward(belt, layer):
        """Belt is the conveyor belt of characters, layers is our learned matrices"""
        outputs = []
        for x in belt:
            up_bias = add_vectors(matrix_vector_multiply(layer["ff_up_weights"], x), layer["ff_up_bias"])
            activated = relu_vector(up_bias)
            down_bias = add_vectors(matrix_vector_multiply(layer["ff_down_weights"], activated), layer["ff_down_bias"])
            outputs.append(down_bias)
        return outputs

    def backwards(all_probabilities, targets, final_belt, intermediate_dicts):
        """Runs the network in reverse, working out how much each weight should change to lower the loss.
        Starts from the loss gradient at the output and walks back the embeddings, returning the gradients."""
        loss_gradient_logits = []

        # Working backwards now.
        # Essentially doing one-hot-like encoding so we know what the right answer was.
        for probability, target in zip(all_probabilities, targets):
            p = probability.copy()
            p[target] = p[target] - 1
            loss_gradient_logits.append(p)

        # Accumulator vector. We keep adding to this.
        output_bias_gradients = [0.0] * vocab_size

        # We loop through all the loss gradient logits for each position and smush them together.
        # This uses loss_gradient_logits, i.e. what the correct 'answer' was.
        for position_gradient in loss_gradient_logits:
            output_bias_gradients = add_vectors(output_bias_gradients, position_gradient)

        # Accumulator matrix of all zeros. We also keep adding to this.
        hidden_to_output_weight_gradient = [[0.0] * EMBEDDING_DIM for _ in range(vocab_size)]

        # Looping through matrix and final_belt
        for weights, belt_vector in zip(loss_gradient_logits, final_belt):
            hidden_to_output_weight_gradient = add_matrices(hidden_to_output_weight_gradient, outer_product(weights, belt_vector))

        # Transposing the weights we just accumulated
        transposed_output_weights = transpose_matrix(hidden_to_output_weights)

        # Send the gradient backwards into the belt, ready to travel down through the blocks
        belt_gradients = [matrix_vector_multiply(transposed_output_weights, position_gradient) for position_gradient in loss_gradient_logits]

        # Grabbing from the last dictionary entry
        last_pre_ln2_belt = intermediate_dicts[-1]["pre_ln2_belt"]

        last_ln2_gamma = layers[-1]["ln2_gamma"]

        # Setting up holders before the loop
        ln2_gamma_gradient = [0.0] * EMBEDDING_DIM
        ln2_beta_gradient = [0.0] * EMBEDDING_DIM
        ln2_input_gradients = []

        # Filling in all our holders
        for gradient, last_pre_ln2_vector in zip(belt_gradients, last_pre_ln2_belt):
            gamma_gradient, beta_gradient, input_gradient = layer_norm_backward(gradient, last_pre_ln2_vector, last_ln2_gamma)
            ln2_gamma_gradient = add_vectors(ln2_gamma_gradient, gamma_gradient)
            ln2_beta_gradient = add_vectors(ln2_beta_gradient, beta_gradient)
            ln2_input_gradients.append(input_gradient)

        last_post_attention_belt = intermediate_dicts[-1]["post_attention_belt"]

        ff_up_weights_gradient, ff_up_bias_gradient, ff_down_weights_gradient, ff_down_bias_gradient, input_gradients = feed_forward_backward(ln2_input_gradients, last_post_attention_belt, layers[-1])

        # Residual join
        post_attention_belt_gradients = [add_vectors(x, y) for x, y in zip(input_gradients, ln2_input_gradients)]

        # Grabbing from the last dictionary entry
        last_pre_ln1_belt = intermediate_dicts[-1]["pre_ln1_belt"]

        last_ln1_gamma = layers[-1]["ln1_gamma"]

        # Setting up holders before the loop
        ln1_gamma_gradient = [0.0] * EMBEDDING_DIM
        ln1_beta_gradient = [0.0] * EMBEDDING_DIM
        ln1_input_gradients = []

        # Filling in all our holders
        for gradient, last_pre_ln1_vector in zip(post_attention_belt_gradients, last_pre_ln1_belt):
            gamma_gradient, beta_gradient, input_gradient = layer_norm_backward(gradient, last_pre_ln1_vector, last_ln1_gamma)
            ln1_gamma_gradient = add_vectors(ln1_gamma_gradient, gamma_gradient)
            ln1_beta_gradient = add_vectors(ln1_beta_gradient, beta_gradient)
            ln1_input_gradients.append(input_gradient)

        return loss_gradient_logits

    def transformer_block(belt, layer):
        attention_out = multi_head_attention(belt, layer)
        intermediate_dict = {}

        pre_ln1_belt = [add_vectors(x, y) for x, y in zip(belt, attention_out)]
        intermediate_dict["pre_ln1_belt"] = pre_ln1_belt

        post_attention_belt = [layer_norm(pre_ln1_belt, layer["ln1_gamma"], layer["ln1_beta"]) for x, y in zip(belt, attention_out)]
        feed_forward_output = feed_forward(post_attention_belt, layer)

        # Calculating the intermediate step here, as we need to pass intermediates to forward.
        pre_ln2_belt = [add_vectors(x, y) for x, y in zip(post_attention_belt, feed_forward_output)]
        intermediate_dict["pre_ln2_belt"] = pre_ln2_belt
        intermediate_dict["post_attention_belt"] = post_attention_belt

        # This is the post-ln2 stuff.
        final_list = [layer_norm(x, layer["ln2_gamma"], layer["ln2_beta"]) for x in pre_ln2_belt]

        return final_list, intermediate_dict

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

    def layer_norm(vector, gamma, beta):
        """Works across the features of a single position's vector"""
        vector_length = len(vector)

        # Calculating the mean
        mean = sum(vector) / vector_length

        # Calculating the variance
        variance = sum((v - mean) ** 2 for v in vector) / vector_length

        # Calculating standard deviation
        standard_dev = (variance + LAYER_NORM_EPS) ** 0.5

        normalised_vector = [(v - mean) / standard_dev for v in vector]

        return [gamma[i] * normalised_vector[i] + beta[i] for i in range(vector_length)]

    def layer_norm_backward(gradient, layer_norm_input, gamma):
        """The reverse of the layer_norm function above, for backprop"""

        # Same going out as in
        beta_gradient = gradient

        normalised = layer_norm(layer_norm_input, [1.0] * len(layer_norm_input), [0.0] * len(layer_norm_input))
        gamma_gradient = multiply_vectors_elementwise(gradient, normalised)

        normalised_gradient = multiply_vectors_elementwise(gradient, gamma)

        # Getting the standard deviation of the layer_norm_input
        vector_length = len(layer_norm_input)
        mean = sum(layer_norm_input) / vector_length
        variance = sum((v - mean) ** 2 for v in layer_norm_input) / vector_length
        standard_dev = (variance + LAYER_NORM_EPS) ** 0.5

        mean_normalised_gradient = sum(normalised_gradient) / vector_length
        mean_gradient_times_normalised = sum(multiply_vectors_elementwise(normalised_gradient, normalised_gradient)) / vector_length

        input_gradient = [(x - mean_normalised_gradient - y * mean_gradient_times_normalised) / standard_dev for x, y in zip(normalised_gradient, normalised)]

        return gamma_gradient, beta_gradient, input_gradient

    def feed_forward_backward(output_gradients, feed_forward_input, layer):
        # More holders / accumulators
        ff_up_weights_gradient = [[0.0] * EMBEDDING_DIM for _ in range(FF_HIDDEN_DIM)]
        ff_up_bias_gradient = [0.0] * FF_HIDDEN_DIM
        ff_down_weights_gradient = [[0.0] * FF_HIDDEN_DIM for _ in range(EMBEDDING_DIM)]
        ff_down_bias_gradient = [0.0] * EMBEDDING_DIM
        input_gradients = []

        # It's accumulatin' time
        for output_gradient, input_vector in zip(output_gradients, feed_forward_input):
            up = add_vectors(matrix_vector_multiply(layer["ff_up_weights"], input_vector), layer["ff_up_bias"])
            activated = relu_vector(up)
            ff_down_bias_gradient = add_vectors(ff_down_bias_gradient, output_gradient)
            ff_down_weights_gradient = add_matrices(ff_down_weights_gradient, outer_product(output_gradient, activated))
            activated_gradient = matrix_vector_multiply(transpose_matrix(layer["ff_down_weights"]), output_gradient)

            up_gradient = [x if y > 0 else 0.0 for x, y in zip(activated_gradient, up)]  # type: ignore
            ff_up_bias_gradient = add_vectors(ff_up_bias_gradient, up_gradient)
            ff_up_weights_gradient = add_matrices(ff_up_weights_gradient, outer_product(up_gradient, input_vector))
            input_gradients.append(matrix_vector_multiply(transpose_matrix(layer["ff_up_weights"]), up_gradient))

        return ff_up_weights_gradient, ff_up_bias_gradient, ff_down_weights_gradient, ff_down_bias_gradient, input_gradients

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

    def sample(seed_char_id):
        """Returns a string of generated text of length n_chars"""

        context = [seed_char_id]
        generated_chars = []

        for _ in range(SAMPLE_SIZE):
            # We run forward on the context, returning us a probability distribution for the next character
            all_probabilities, _, _ = forward(context[-SEQUENCE_LENGTH:])
            # We only want the last distribution as the others only helped to get us to the final one
            next_char_probability_distribution = all_probabilities[-1]
            # We randomly select based on the distribution, using probabilities as our weights
            random_char_id = random.choices(range(vocab_size), weights=next_char_probability_distribution)[0]
            # And set up the next loop
            context.append(random_char_id)
            generated_chars.append(id_to_char[random_char_id])

        return "".join(generated_chars)

    def append_loss(iteration, loss):
        """Appends one row to outputs/loss.csv."""
        with open(output_dir / "loss.csv", "a", encoding="utf-8") as f:
            f.write(f"{iteration},{loss}\n")

    def append_sample(iteration, sample_text):
        """Appends a labelled sample block to outputs/samples.txt."""
        with open(output_dir / "samples.txt", "a", encoding="utf-8") as f:
            f.write(f"=== Iteration {iteration} ===\n{sample_text}\n\n")

    seed_char_id = char_to_id["A"]
    print(sample(seed_char_id))


if __name__ == "__main__":
    main()
