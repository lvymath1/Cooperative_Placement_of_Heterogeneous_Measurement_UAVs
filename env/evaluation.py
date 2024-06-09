import numpy as np


def evaluation(A, x, eta, sig, S):
    A_S = A[S]
    MSE = np.trace(np.linalg.inv(np.transpose(A_S)  @ np.diag(sig[S]) @ A_S))
    return 10 * np.log10(MSE)