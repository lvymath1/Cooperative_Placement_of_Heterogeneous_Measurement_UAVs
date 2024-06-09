import numpy as np
from scipy.interpolate import griddata


def large_scale_evaluation(A, x, x_choose, eta, S, points, size_M, size_N):
    y = A @ x_choose + eta
    A_S = A[S]
    x_hat = np.linalg.pinv(A_S) @ y[S]
    x_grid, y_grid = np.meshgrid(np.linspace(0, size_M - 1, size_M), np.linspace(0, size_N - 1, size_N))
    interpolated_temperatures = griddata(points, x_hat, (x_grid, y_grid), method='cubic').ravel()
    np.set_printoptions(threshold=np.inf)
    return np.mean(np.abs(interpolated_temperatures - x))