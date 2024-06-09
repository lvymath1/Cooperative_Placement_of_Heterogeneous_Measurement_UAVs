import numpy as np


def DGA(M1, M2, M3, N1, N2, N3, A, epsilon=1):
    N, K = A.shape
    N = N1 + N2 + N3
    cf = np.zeros(N)
    for j in range(N1):
        cf[j] = np.log10(
            np.linalg.det(A[j, :].reshape(-1, 1) @ A[j, :].reshape(1, -1) + epsilon * np.eye(K)))  # 计算 cf(j)

    ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引

    ind_list = [ind]  # 保存每轮的最大索引

    for _ in range(M1 - 1):
        cf = np.zeros(N)
        for j in range(N1):
            if j not in ind_list:
                cf[j] = np.log(np.linalg.det(
                    A[np.ix_([ind, j], range(A.shape[1]))].T @ A[
                        np.ix_([ind, j], range(A.shape[1]))] + epsilon * np.eye(K)))  # 计算 cf(j)
        cf = np.abs(cf)
        ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引
        ind_list.append(ind)

    for _ in range(M1, M1 + M2):
        cf = np.zeros(N)
        for j in range(N1, N1 + N2):
            if j not in ind_list:
                cf[j] = np.log(np.linalg.det(
                    A[np.ix_([ind, j], range(A.shape[1]))].T @ A[
                        np.ix_([ind, j], range(A.shape[1]))] + epsilon * np.eye(K)))  # 计算 cf(j)
        cf = np.abs(cf)
        ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引
        ind_list.append(ind)
    for _ in range(M1 + M2, M1 + M2 + M3):
        cf = np.zeros(N)
        for j in range(N1 + N2, N1 + N2 + N3):
            if j not in ind_list:
                cf[j] = np.log(np.linalg.det(
                    A[np.ix_([ind, j], range(A.shape[1]))].T @ A[
                        np.ix_([ind, j], range(A.shape[1]))] + epsilon * np.eye(K)))  # 计算 cf(j)
        cf = np.abs(cf)
        ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引
        ind_list.append(ind)
    return ind_list

def DGA2(M1, M2, N1, N2, A, epsilon):
    N, K = A.shape

    cf = np.zeros(N1 + N2)
    for j in range(N1 + N2):
        cf[j] = np.log10(
            np.linalg.det(A[j, :].reshape(-1, 1) @ A[j, :].reshape(1, -1) + epsilon * np.eye(K)))  # 计算 cf(j)

    ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引

    ind_list = [ind]  # 保存每轮的最大索引
    for i in range(M1):
        cf = np.zeros(N1)
        for j in range(N1):
            if j not in ind_list:
                cf[j] = np.log(np.linalg.det(
                    A[np.ix_([ind, j], range(A.shape[1]))].T @ A[
                        np.ix_([ind, j], range(A.shape[1]))] + epsilon * np.eye(K)))  # 计算 cf(j)
        cf = np.abs(cf)
        ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引
        ind_list.append(ind)
    for i in range(M1, M1 + M2):
        cf = np.zeros(N2)
        for j in range(N1, N1 + N2):
            if j not in ind_list:
                cf[j - N1] = np.log(np.linalg.det(
                    A[np.ix_([ind, j], range(A.shape[1]))].T @ A[
                        np.ix_([ind, j], range(A.shape[1]))] + epsilon * np.eye(K)))  # 计算 cf(j)
        cf = np.abs(cf)
        ind = np.argmax(cf)  # 找到 cf 中的最大值对应的索引
        ind_list.append(ind + N1)
    return ind_list