import numpy as np


def JGA(M1, M2, M3, N1, N2, N3, A, sig):
    N = N1 + N2 + N3
    select1 = list(range(N1))
    select2 = list(range(N1, N1 + N2))
    select3 = list(range(N1 + N2, N1 + N2 + N3))
    # Normalize the norm of the rows
    for i in range(N):
        A[i, :] = A[i, :] / np.linalg.norm(A[i, :], ord=2)

    # Greedy alg
    # First Step, find the best two rows
    # G = A @ A.T @ np.diag(sig_1) @ np.diag(sig_1) @ A @ A.T
    G = A @ A.T @ np.diag(sig) @ np.diag(sig) @ A @ A.T
    # np.fill_diagonal(G, 0)
    Gsum = np.sum(G, axis=0)

    # 一起消除
    while len(select1) > M1 or len(select2) > M2 or len(select3) > M3:
        select = []
        if len(select1) > M1:
            select += select1
        if len(select2) > M2:
            select += select2
        if len(select3) > M3:
            select += select3
        ind = np.argmax(Gsum[select])
        Gsum = Gsum - G[select[ind], :]
        Gsum[select[ind]] = -np.inf
        if select[ind] < N1:
            select1.remove(select[ind])
        elif select[ind] < N1 + N2:
            select2.remove(select[ind])
        else:
            select3.remove(select[ind])
    return select1 + select2 + select3


def JGA2(M1, M2, N1, N2, A, sig):
    N = N1 + N2
    select1 = list(range(N1))
    select2 = list(range(N1, N1 + N2))
    sig_mean = np.mean(sig)
    sig_1 = np.reciprocal(1 + np.exp(-(sig - sig_mean)))
    # Normalize the norm of the rows
    for i in range(N):
        A[i, :] = A[i, :] / np.linalg.norm(A[i, :], ord=2)

    # Greedy algorithm
    # First Step, find the best two rows
    G = A @ A.T @ np.diag(sig_1) @ np.diag(sig_1) @ A @ A.T
    # np.fill_diagonal(G, 0)
    Gsum = np.sum(G, axis=0)

    # Find the row to eliminate one by one
    while len(select1) > M1 or len(select2) > M2:
        select = []
        if len(select1) > M1:
            select += select1
        if len(select2) > M2:
            select += select2
        ind = np.argmax(Gsum[select])
        Gsum = Gsum - G[select[ind], :]
        Gsum[select[ind]] = -np.inf
        if select[ind] < N1:
            select1.remove(select[ind])
        else:
            select2.remove(select[ind])
    return select1 + select2