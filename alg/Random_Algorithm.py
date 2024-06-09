import numpy as np


def RA(M1, M2, M3, N1, N2, N3):
    S1 = np.random.choice(range(N1), size=M1, replace=False)
    S2 = np.concatenate((S1, np.random.choice(range(N1, N1 + N2), size=M2, replace=False)))
    S3 = np.concatenate((S2, np.random.choice(range(N1 + N2, N1 + N2 + N3), size=M3, replace=False)))
    return S3

def RA2(M1, M2, N1, N2):
    return list(np.concatenate((np.random.choice(range(N1), size = M1, replace=False), np.random.choice(range(N1, N1 + N2), size = M2, replace=False))))