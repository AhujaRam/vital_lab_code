
import numpy as np
from scipy.spatial import distance
import pandas as pd
import math

def activity_vector_rdm(activity_vectors):
    num_vectors = len(activity_vectors)
    dissimilarity_matrix = np.full((num_vectors, num_vectors), np.nan)

    for i in range(num_vectors):
        dissimilarity_matrix[i, i] = 0.0  # Set the diagonal values to 0

        for j in range(i + 1, num_vectors):
            correlation = np.corrcoef(activity_vectors[i], activity_vectors[j])[0, 1]
            dissimilarity = 1.0 - correlation
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity

    return dissimilarity_matrix

