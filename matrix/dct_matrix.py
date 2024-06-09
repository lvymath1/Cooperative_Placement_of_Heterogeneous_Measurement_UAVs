import numpy as np
from scipy.fft import dct


def generate_dct_matrix(N, K):
    random_matrix = np.random.rand(N, K)
    # 计算DCT变换
    dct_matrix = dct(random_matrix, axis=1, norm='ortho')
    return dct_matrix