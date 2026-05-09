"""
Cleans Shakespeare input data and trains Word2Vec embeddings.
After training, embeddings + vocabulary are saved to disk; use query.py to explore.
"""

import random
import math
import time
import collections
import bisect
from pathlib import Path
import numpy as np

# Constants
CONTEXT_WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 5
EMBEDDING_SIZE = 50
LEARNING_RATE = 0.025
EPOCHS = 50
SH_WORDS_READ = 100000
SUBSAMPLE_THRESHOLD = 1e-3 # Threshold from original Word2Vec paper

SCRIPT_DIR = Path(__file__).parent

# Simple sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A method to strip out all of the junk characters we don't want to be dealing with.
def strip_junk(text):
    text = ' '.join("".join(ch for ch in text if ch.isalpha() or ch == " ").split())

    return text

# Returns only the positive pairs to help reinforce correct word embeddings
def find_positive_pairs(id_list: list[int]) -> list[tuple[int, int]]:
    positive_pairs = []
    n = len(id_list)
    for i in range(n):

        start = max(0, i - CONTEXT_WINDOW_SIZE)
        end   = min(n, i + CONTEXT_WINDOW_SIZE + 1)

        for j in range(start, end):
            if i != j:
                positive_pairs.append((id_list[i], id_list[j]))
    return(positive_pairs)

# Adds negative pairs to the training data
def add_negative_pairs(positive_pair, cumulative_unigram):
    c_id = positive_pair[0]
    p_id = positive_pair[1]

    negative_examples = []

    while len(negative_examples) < NEGATIVE_SAMPLES:
        random_number = random.random()
        index = bisect.bisect_left(cumulative_unigram, random_number)

        if index == p_id or index in negative_examples:
            continue
        negative_examples.append(index)

    return [c_id, p_id, negative_examples]

# Runs training once on a single center word
def train_single_example(center_id: int, pos_id: int, neg_ids: list[int], embeddings_in, embeddings_out):

    other_ids = [pos_id] + neg_ids
    others = embeddings_out[other_ids]
    center_vec = embeddings_in[center_id]

    dot_products = others @ center_vec
    scores = sigmoid(dot_products)

    # Simple way to gen all required labels (1 positive, then 5 zeroes for the negatives)
    labels = np.array([1.0] + [0.0] * len(neg_ids))

    errors = scores - labels

    # Using fancy numpy vector matrix operations to speed up training
    gradient_others = errors[:, None] * center_vec
    gradient_center = errors @ others

    center_vec -= LEARNING_RATE * gradient_center
    embeddings_out[other_ids] -= LEARNING_RATE * gradient_others

# Runs training for one full cycle across all given examples
def train_one_epoch(training_data, embeddings_in, embeddings_out):
    for c_id, p_id, neg_ids in training_data:
        train_single_example(c_id, p_id, neg_ids, embeddings_in, embeddings_out)


def main():
    # Bringing in corpus data
    shakespeare_file = SCRIPT_DIR / "Original Data" / "Shakespeare.txt"

    # All corpus text lower-case for better matching
    shakespeare_text = open(shakespeare_file).read().lower()

    # Keeps letters and single spaces, splits corpus into list of words
    shakespeare_word_list = "".join(ch for ch in shakespeare_text if ch.isalpha() or ch == " ").split()[0:SH_WORDS_READ]
    shakespeare_word_count_dictionary = collections.Counter(shakespeare_word_list)
    unique_vocabulary = [item[0] for item in shakespeare_word_count_dictionary]

    # Building out a dictionary for our vocabulary, allowing a reverse dict lookup.
    # We use this over a Word class for speed/memory reasons. It's a NN thing it seems.
    word_to_id = {}

    for index in range(len(unique_vocabulary)):
        word = unique_vocabulary[index]
        word_to_id[word] = index

    # Looking to reduce instances of frequent filler words such as 'to' or 'as' as these can pollute results.
    # Dictionary for us to assess whether we keep each word - stochastic downsampling approach.
    
    keep_prob = {}
    for word, count in shakespeare_word_count_dictionary.items():
        # Assessing how often eaach word appears in our vocabulary
        freq_ratio = count / len(shakespeare_word_list)

        # Some instances of filler words are kept - to a ratio defined by the threshold
        keep_prob[word] = min(1.0, math.sqrt(SUBSAMPLE_THRESHOLD / freq_ratio))

    # Retaining the filtered word instances but as IDs - random lets us select roughly the right number
    filtered_ids = [word_to_id[word] for word in shakespeare_word_list if random.random() < keep_prob[word]]

    # We generate a list of words that should be associated given their proximity in Shakespeare
    positive_pairs = find_positive_pairs(filtered_ids)

    # Building the cumulative unigram distribution for negative sampling.
    # Using the 0.75 power trick from Word2Vec which worked the best for them.
    # This attempts to train a line when randomly selecting negative words - between per-frequency occurrences or from unique word list
    freq_pow = np.array([shakespeare_word_count_dictionary[w] for w in unique_vocabulary]) ** 0.75
    cumulative_unigram = np.cumsum(freq_pow / freq_pow.sum())

    # We generate negative data so we can ensure they are NOT associated, as per Shakespeare text
    training_data = [add_negative_pairs(x, cumulative_unigram) for x in positive_pairs]

    # Initialising fully random embeddings.
    # Using two matrices to remove pollution caused by each word being either context or pos/neg examples.
    embeddings_in = np.random.uniform(-0.01, 0.01, size = (len(unique_vocabulary), EMBEDDING_SIZE))
    embeddings_out = np.random.uniform(-0.01, 0.01, size = (len(unique_vocabulary), EMBEDDING_SIZE))

    print(f"Starting training on {len(training_data)} examples")

    # We'll iterate and train the embeddings on pos and neg examples over each epoch.
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Apparently this is standard practice.
        random.shuffle(training_data)
        train_one_epoch(training_data, embeddings_in, embeddings_out)
        print(f"Epoch {epoch + 1} done in {round(time.time() - epoch_start)} seconds")

    np.save("embeddings_in.npy", embeddings_in)
    np.save("embeddings_out.npy", embeddings_out)

    with open("vocabulary.txt", "w") as f:
        f.write("\n".join(unique_vocabulary))

    print("Saved embeddings + vocabulary. Use query.py to explore the trained model.")

if __name__ == "__main__":
    main()
