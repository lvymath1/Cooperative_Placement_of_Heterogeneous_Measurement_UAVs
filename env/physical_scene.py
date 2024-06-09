import numpy as np

from matrix.dct_matrix import generate_dct_matrix
from matrix.low_correlation_random_matrix import generate_low_correlation_random_matrix
from matrix.orth_matrix import generate_orth_matrix


def physical_scene(N1, N2, N3, sig1, sig2, sig3, K, matrix):
    N = N1 + N2 + N3

    # Dictionary to map matrix types to their corresponding functions
    matrix_generators = {
        "Dct matrix": generate_dct_matrix,
        "Orthogonal matrix": generate_orth_matrix,
        "Low correlation matrix": generate_low_correlation_random_matrix
    }
    # Select the appropriate matrix generation function
    generate_matrix = matrix_generators.get(matrix)
    if not generate_matrix:
        raise ValueError(f"Unknown matrix type: {matrix}")

    # Generate the matrix
    A = generate_matrix(N, K)

    # Generate the random vector x
    x = np.random.rand(K)

    # Generate the noise vector eta
    eta = np.concatenate([
        sig1 * np.random.randn(N1),
        sig2 * np.random.randn(N2),
        sig3 * np.random.randn(N3)
    ])

    # Generate the standard deviation vector sig
    sig = np.array([sig1] * N1 + [sig2] * N2 + [sig3] * N3)

    return A, x, eta, sig