import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def plot_fire_scene(s, A, eta, x_choose, S, size_M, size_N, points):
    y = A @ x_choose + eta
    A_S = A[S]
    x_hat = np.linalg.pinv(A_S) @ y[S]
    x_grid, y_grid = np.meshgrid(np.linspace(0, size_M - 1, size_M), np.linspace(0, size_N - 1, size_N))
    interpolated_temperatures = griddata(points, x_hat, (x_grid, y_grid), method='cubic')
    # 创建热力图
    plt.imshow(interpolated_temperatures, cmap="YlOrRd", extent=[0, size_M - 1, 0, size_N - 1], origin='lower', aspect='auto')
    plt.xlim(0, 99)  # 设置x轴范围
    plt.ylim(0, 199)  # 设置y轴范围
    x_ticks = np.linspace(0, size_M - 1, num=9)  # 生成横坐标刻度
    y_ticks = np.linspace(0, size_N - 1, num=9)  # 生成纵坐标刻度
    plt.xticks(x_ticks, np.linspace(0, 1, num=9))  # 将刻度对应到0~0.99
    plt.yticks(y_ticks, np.linspace(0, 2, num=9))  # 将刻度对应到0~1.99
    plt.colorbar(label='Temperature(℃)')
    plt.title('Heatmap:' + s, fontsize=22)
    plt.xlabel('X-axis (km)', fontsize=18)
    plt.ylabel('Y-axis (km)', fontsize=18)