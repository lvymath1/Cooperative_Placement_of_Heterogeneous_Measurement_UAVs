import numpy as np
from scipy.linalg import orth


def generate_orth_matrix(N, K):
    matrix = np.random.random((N, N))
    orthogonal_matrix = orth(matrix)
    return orthogonal_matrix[:, :K]