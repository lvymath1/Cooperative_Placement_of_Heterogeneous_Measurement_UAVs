import numpy as np
from scipy.linalg import orth


# 生成随机矩阵
def generate_low_correlation_random_matrix(rows, cols, noise_level=0.1):
    random_matrix = np.random.rand(rows, cols)

    orthogonal_matrix = orth(random_matrix)

    noise = noise_level * np.random.rand(rows, cols)
    low_correlation_matrix = orthogonal_matrix + noise

    norms = np.linalg.norm(low_correlation_matrix, axis=0)
    low_correlation_matrix = low_correlation_matrix / norms

    return low_correlation_matrix
