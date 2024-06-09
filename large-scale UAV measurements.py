import time
import numpy as np
from matplotlib import pyplot as plt

from alg.Determinant_Algorithm import DGA2
from alg.Joint_Greedy_Algorithm import JGA2
from alg.Random_Algorithm import RA2
from alg.Random_Greedy_Algorithm import RGA2
from env.evaluation import evaluation
from env.large_scale_physical_scene import large_scale_physical_scene
from plot.plot_fire_scene import plot_fire_scene

np.random.seed(963)

# Define the problem size and parameters
size_M, size_N = 100, 200
choose = 65
N1, N2 = 80, 200
M1, M2 = 20, 55

# Generate the physical scene data
A, x, x_choose, eta, sig, point = large_scale_physical_scene(size_M, size_N, N1, N2, sig1=2, sig2=6, K=choose)

# Execute the Random Greedy Algorithm (RGA2)
start_time = time.time()
S_RGA = RGA2(M1, M2, N1, N2, A, sig)
end_time = time.time()
print("RGA runtime:", end_time - start_time)
print("S_RGA:", S_RGA)

# Execute the Joint Greedy Algorithm (JGA2)
start_time = time.time()
S_JGA = JGA2(M1, M2, N1, N2, A, sig)
end_time = time.time()
print("JGA runtime:", end_time - start_time)
print("S_JGA:", S_JGA)

# Execute the Determinant Greedy Algorithm (DGA2)
start_time = time.time()
S_DGA = DGA2(M1, M2, N1, N2, A, 1)
end_time = time.time()
print("DGA runtime:", end_time - start_time)
print("S_DGA:", S_DGA)

# Execute the Random Algorithm (RA2)
start_time = time.time()
S_random = RA2(M1, M2, N1, N2)
end_time = time.time()
print("Random algorithm runtime:", end_time - start_time)
print("S_random:", S_random)

# Compute Mean Squared Error (MSE) for each solution
print("S_RGA evaluation MSE:", evaluation(A, x_choose, eta, sig, S_RGA))
print("S_JGA evaluation MSE:", evaluation(A, x_choose, eta, sig, S_JGA))
print("S_DGA evaluation MSE:", evaluation(A, x_choose, eta, sig, S_DGA))
print("S_RA evaluation MSE:", evaluation(A, x_choose, eta, sig, S_random))

# Plot the fire scenes for each algorithm
plt.rcParams['font.family'] = 'Microsoft YaHei'
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

plt.subplot(2, 2, 1)
plot_fire_scene("RGA", A, eta, x_choose, S_RGA, size_M, size_N, point)

plt.subplot(2, 2, 2)
plot_fire_scene("JGA", A, eta, x_choose, S_JGA, size_M, size_N, point)

plt.subplot(2, 2, 3)
plot_fire_scene("DGA", A, eta, x_choose, S_DGA, size_M, size_N, point)

plt.subplot(2, 2, 4)
plot_fire_scene("RA", A, eta, x_choose, S_random, size_M, size_N, point)

plt.tight_layout()
plt.savefig("fig/temperature.jpg")
