import numpy as np

# Returns the dot-product between two vectors. Higher = more aligned in direction AND magnitude.
def DotProduct(v1, v2):
    return np.dot(v1, v2)

# Cosine similarity ignores magnitude and returns just the angle between two vectors.
# Higher (closer to 1) = more similar in direction.
def CosineSimilarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Straight-line distance between two vectors. Lower = more similar (closer in space).
def EuclideanDistance(v1, v2):
    return np.linalg.norm(v1 - v2)
