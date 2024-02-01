def activity_vector_rdm(activity_vectors):
    num_vectors = len(activity_vectors)
    dissimilarity_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            # Compute Pearson correlation coefficient
            correlation = np.corrcoef(activity_vectors[i], activity_vectors[j])[0, 1]

            # Calculate correlation dissimilarity (1 - correlation)
            dissimilarity = 1.0 - correlation

            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity

    return dissimilarity_matrix

