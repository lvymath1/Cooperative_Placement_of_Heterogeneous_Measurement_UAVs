import numpy as np


def RGA(M1, M2, M3, N1, N2, N3, A, sig):
    N = N1 + N2 + N3
    select1 = list(range(N1))
    select2 = list(range(N1, N1 + N2))
    select3 = list(range(N1 + N2, N1 + N2 + N3))

    # Normalize the norm of the rows
    for i in range(N):
        A[i, :] = A[i, :] / np.linalg.norm(A[i, :], ord=2)

    # Greedy algorithm
    # First Step, find the best two rows
    # G = A @ A.T @ np.diag(sig) @ np.diag(sig) @ A @ A.T
    G = A @ A.T @ np.diag(sig) @ np.diag(sig) @ A @ A.T
    Gsum = np.sum(G, axis=1)

    choices = [1] * (N1 - M1) + [2] * (N2 - M2) + [3] * (N3 - M3)
    np.random.shuffle(choices)

    for choice in choices:
        if choice == 1:
            ind = np.argmax(Gsum[select1])
            Gsum = Gsum - G[select1[ind], :]
            select1.remove(select1[ind])
        if choice == 2:
            ind = np.argmax(Gsum[select2])
            Gsum = Gsum - G[select2[ind], :]
            select2.remove(select2[ind])
        if choice == 3:
            ind = np.argmax(Gsum[select3])
            Gsum = Gsum - G[select3[ind], :]
            select3.remove(select3[ind])
    return select1 + select2 + select3


def RGA2(M1, M2, N1, N2, A, sig):
    N = N1 + N2
    select1 = list(range(N1))
    select2 = list(range(N1, N1 + N2))

    # Normalize the norm of the rows
    for i in range(N):
        A[i, :] = A[i, :] / np.linalg.norm(A[i, :], ord=2)

    # Greedy algorithm
    # First Step, find the best two rows
    G = A @ A.T @  np.diag(sig) @ np.diag(sig) @ A @ A.T
    # np.fill_diagonal(G, 0)
    Gsum = np.sum(G, axis=0)

    # Find the row to eliminate one by one
    choices = [1] * (N1 - M1) + [2] * (N2 - M2)
    np.random.shuffle(choices)

    for choice in choices:
        if choice == 1:
            ind = np.argmax(Gsum[select1])
            Gsum = Gsum - G[select1[ind], :]
            select1.remove(select1[ind])
        if choice == 2:
            ind = np.argmax(Gsum[select2])
            Gsum = Gsum - G[select2[ind], :]
            select2.remove(select2[ind])
    return select1 + select2