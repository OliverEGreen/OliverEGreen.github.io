import numpy as np

from embedding_helpers import DotProduct, CosineSimilarity, EuclideanDistance

# Load the trained embeddings + vocabulary saved by Cleaning Raw Data.py.
embeddings_in = np.load("embeddings_in.npy")
vocabulary = open("vocabulary.txt").read().splitlines()
word_to_id = {word: idx for idx, word in enumerate(vocabulary)}

# Assesses how similar two words are by running DotProduct over their vectors.
def WordSimilarity(word_1, word_2):
    if word_1 not in word_to_id:
        return f"Word {word_1} could not be found in the vocabulary dictionary."
    if word_2 not in word_to_id:
        return f"Word {word_2} could not be found in the vocabulary dictionary."
    id_1 = word_to_id[word_1]
    id_2 = word_to_id[word_2]
    vector_1 = embeddings_in[id_1]
    vector_2 = embeddings_in[id_2]
    return DotProduct(vector_1, vector_2)

def FindMostSimilarWords(target_word: str, method = "DotProduct"):
    if target_word not in word_to_id:
        return f"{target_word} was not found in the vocabulary dictionary."
    target_id = word_to_id[target_word]
    target_vector = embeddings_in[target_id]
    scores = []

    for word_id, word in enumerate(vocabulary):
        if word == target_word:
            continue
        if method == "DotProduct":
            dot_product = DotProduct(target_vector, embeddings_in[word_id])
            scores.append((word, dot_product))
        elif method == "CosineSimilarity":
            cos_similarity = CosineSimilarity(target_vector, embeddings_in[word_id])
            scores.append((word, cos_similarity))
        elif method == "EuclideanDistance":
            euclidean_distance = EuclideanDistance(target_vector, embeddings_in[word_id])
            scores.append((word, euclidean_distance))

    # Euclidean distance wants closer/lower values, other two want higher scores
    scores.sort(key=lambda x: x[1], reverse = method != "EuclideanDistance")
    return scores[:5]

def PrintMostSimilarWordsReport(word: str, method: str):
    print(f"Using {method} as similarity measure.\nMost similar to {word}: {FindMostSimilarWords(word, method)}")
