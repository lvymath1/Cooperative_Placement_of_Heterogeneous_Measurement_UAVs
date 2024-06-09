import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from matrix.orth_matrix import generate_orth_matrix

def generate_x(size_M, size_N):
    # 生成随机数据
    low_temperature = 30
    high_temperature = 400
    data = np.random.uniform(low_temperature, high_temperature, size=(4, 4))
    # 创建插值函数
    x = np.linspace(0, 8, 4)
    y = np.linspace(0, 8, 4)
    # 更多点用于插值，实现平滑效果
    x_new1 = np.linspace(0, 8, 17)
    y_new1 = np.linspace(0, 8, 17)
    f1 = interpolate.interp2d(x, y, data, kind='linear')
    smooth_data1 = f1(x_new1, y_new1)
    # 更多点用于插值，实现平滑效果
    x_new = np.linspace(0, 8, size_M)
    y_new = np.linspace(0, 8, size_N)
    f = interpolate.interp2d(x_new1, y_new1, smooth_data1, kind='linear')
    # 对新网格进行插值
    smooth_data = f(x_new, y_new)
    plt.plot(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.imshow(smooth_data, cmap="YlOrRd", extent=[0, size_M - 1, 0, size_N - 1], origin='lower',
               aspect='auto')
    plt.xlim(0, 99)  # 设置x轴范围
    plt.ylim(0, 199)  # 设置y轴范围
    x_ticks = np.linspace(0, size_M - 1, num=9)  # 生成横坐标刻度
    y_ticks = np.linspace(0, size_N - 1, num=9)  # 生成纵坐标刻度
    plt.xticks(x_ticks, np.linspace(0, 1, num=9))  # 将刻度对应到0~0.99
    plt.yticks(y_ticks, np.linspace(0, 2, num=9))  # 将刻度对应到0~1.99
    plt.colorbar(label='Temperature(℃)')
    plt.title("Heatmap:" + "Original fire image", fontsize=20)
    plt.xlabel('X-axis (km)', fontsize=20)
    plt.ylabel('Y-axis (km)', fontsize=20)
    plt.savefig("fig/original_fire.jpg")
    return smooth_data.flatten()

def large_scale_physical_scene(size_M, size_N, N1, N2, sig1, sig2, K):
    N = N1 + N2
    A = generate_orth_matrix(N, K)  # 构造观测矩阵
    x = generate_x(size_M, size_N)  # 构造全部物理场景向量
    selected_idx = [0, size_M - 1, size_M * size_N - size_M, size_M * size_N - 1]
    remaining_points = np.random.choice(np.setdiff1d(np.arange(0, size_M * size_N - 1), selected_idx), K - 4,
                                        replace=False)
    selected_idx.extend(remaining_points)
    x_choose = x[selected_idx]
    points = []
    for number in selected_idx:
        points.append([number % size_M, number // size_M])
    points = np.array(points)
    eta = np.concatenate((sig1 * np.random.randn(N1), sig2 * np.random.randn(N2)))
    sig = np.array([sig1] * N1 + [sig2] * N2)
    return A, x, x_choose, eta, sig, points