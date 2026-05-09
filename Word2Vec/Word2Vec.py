import random
import math
import time
import bisect
import numpy as np

# Cleans our raw Shakespeare input data and trains Word2Vec embeddings on it.
# After training, embeddings + vocabulary are saved to disk; use query.py to explore.

# Constants
CONTEXT_WINDOW_SIZE = 5
EMBEDDING_SIZE = 50
LEARNING_RATE = 0.025
EPOCHS = 50
SH_WORDS_READ = 100000
SUBSAMPLE_THRESHOLD = 1e-3 # Threshold from original Word2Vec paper

# Reads a file
def GetFileText(file):
    return open(file).read()

# Forces a string to be all lowercase
def LowerCase(text):
    return text.lower()

# Simple sigmoid function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# A method to strip out all of the junk characters we don't want to be dealing with.
def StripJunk(text):
    # Strips out all non-alphabetical characters from a string
    def ReturnOnlyChars(text):
        output_text = ""
        for i in range(0, len(text)):
            if text[i] == " ":
                output_text += text[i]
                continue
            if text[i].isalpha():
                output_text += text[i]
                continue
            output_text += " "
        return output_text

    # Removes any multiple whitespaces from a string
    def StripMultipleSpaces(text):
        return ' '.join(text.split())

    text = ReturnOnlyChars(text)
    text = StripMultipleSpaces(text)

    return text

# Splits a long string into a list of per-word strings
def CreateWordList(text):
    return text.split()

# Creates a dictionary to count unique words in a list of strings
def CountWords(word_list):
    dictionary = {}

    for i in range(0, len(word_list)):
        if word_list[i] in dictionary:
            dictionary[word_list[i]] += 1
            continue
        dictionary[word_list[i]] = 1
    return dictionary

# Returns only the positive pairs to help reinforce correct word embeddings
def FindPositivePairs(word_list: list[str]) -> list[list[int]]:
    positive_pairs = []
    n = len(word_list)
    for i in range(n):

        start = max(0, i - CONTEXT_WINDOW_SIZE)
        end   = min(n, i + CONTEXT_WINDOW_SIZE + 1)

        for j in range(start, end):
            if i != j:
                positive_pairs.append((i, j))
    return(positive_pairs)

# Adds negative pairs to the training data
def AddNegativePairs(positive_pair, word_list, word_to_id, cumulative_unigram, vocabulary):
    # Grabbing position of words from vocab dict
    c_pos = positive_pair[0]
    p_pos = positive_pair[1]

    # Grabbing the words themselves
    center_word = word_list[c_pos]
    pos_word = word_list[p_pos]

    c_id = word_to_id[center_word]
    p_id = word_to_id[pos_word]

    negative_examples = []

    while len(negative_examples) < 5:
        random_number = random.random()
        index = bisect.bisect_left(cumulative_unigram, random_number)
        random_word = vocabulary[index]

        if random_word == pos_word or random_word in negative_examples:
            continue
        negative_examples.append(word_to_id[random_word])

    return [c_id, p_id, negative_examples]

def TrainSingleExample(center_id: int, pos_id: int, neg_ids: list[int], embeddings_in, embeddings_out):

    other_ids = [pos_id] + neg_ids
    others = embeddings_out[other_ids]
    center_vec = embeddings_in[center_id]

    dot_products = others @ center_vec
    scores = Sigmoid(dot_products)

    # Simple way to gen all required labels (1 positive, then 5 zeroes for the negatives)
    labels = np.array([1.0] + [0.0] * len(neg_ids))

    errors = scores - labels

    # Using fancy numpy vector matrix operations to speed up training
    gradient_others = errors[:, None] * center_vec
    gradient_center = errors @ others

    center_vec -= LEARNING_RATE * gradient_center
    embeddings_out[other_ids] -= LEARNING_RATE * gradient_others

# Runs training for one full cycle across all given examples
def TrainOneEpoch(training_data, embeddings_in, embeddings_out):
    for c_id, p_id, neg_ids in training_data:
        TrainSingleExample(c_id, p_id, neg_ids, embeddings_in, embeddings_out)

sh_file = '/Users/olivergreen/Documents/Training LLM/Original Data/Shakespeare.txt'

sh_text = GetFileText(sh_file)
lower_sh_text = LowerCase(sh_text)
stripped_sh_text = StripJunk(lower_sh_text)
sh_word_list = CreateWordList(stripped_sh_text)[0:SH_WORDS_READ]
sh_dict = CountWords(sh_word_list)
sh_sorted_tuples = sorted(sh_dict.items(), key = lambda item: item[1], reverse = True)
vocabulary = [item[0] for item in sh_sorted_tuples]

# Building out a dictionary for our vocabulary.
word_to_id = {}

for idx in range(len(vocabulary)):
    word = vocabulary[idx]
    word_to_id[word] = idx

# Building the unigram distribution for negative sampling.
# Using the 0.75 power trick from Word2Vec which worked the best for them.
unigram = []
for word in vocabulary:
    count_pow = sh_dict[word] ** 0.75
    unigram.append(count_pow)

norm = 0
for u in unigram:
    norm = norm + u

for i in range(len(unigram)):
    unigram[i] = unigram[i] / norm

# Looking to filter out frequent filler words such as 'to' or 'as' as these can pollute results.

total_tokens = len(sh_word_list)

# Dictionary for us to assess whether we keep each word. Filtering out the most common words as they're basically noise.
keep_prob = {}
for word, count in sh_dict.items():
    freq_ratio = count / total_tokens
    keep_prob[word] = min(1.0, math.sqrt(SUBSAMPLE_THRESHOLD / freq_ratio))

filtered_word_list = []

for word in sh_word_list:
    if random.random() < keep_prob[word]:
        filtered_word_list.append(word)

# We generate a list of words that should be associated given their proximity in Shakespeare
positive_pairs = FindPositivePairs(filtered_word_list)

cumulative_unigram = []
running_total = 0.0
for u in unigram:
    running_total += u
    cumulative_unigram.append(running_total)

# We generate negative data so we can ensure they are NOT associated, as per Shakespeare text
training_data = [AddNegativePairs(list(x), filtered_word_list, word_to_id, cumulative_unigram, vocabulary) for x in positive_pairs]

# Initialising fully random embeddings.
# Using two matrices to remove pollution caused by each word being either context or pos/neg examples.
embeddings_in = np.random.uniform(-0.01, 0.01, size = (len(vocabulary), EMBEDDING_SIZE))
embeddings_out = np.random.uniform(-0.01, 0.01, size = (len(vocabulary), EMBEDDING_SIZE))

print(f"Starting training on {len(training_data)} examples")

# We'll iterate and train the embeddings on pos and neg examples over each epoch.
for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Apparently this is standard practice.
    random.shuffle(training_data)
    TrainOneEpoch(training_data, embeddings_in, embeddings_out)
    print(f"Epoch {epoch + 1} done in {round(time.time() - epoch_start)} seconds")

np.save("embeddings_in.npy", embeddings_in)
np.save("embeddings_out.npy", embeddings_out)

with open("vocabulary.txt", "w") as f:
    f.write("\n".join(vocabulary))

print("Saved embeddings + vocabulary. Use query.py to explore the trained model.")
