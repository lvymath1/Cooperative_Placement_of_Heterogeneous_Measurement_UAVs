import time
import numpy as np
from alg.Determinant_Algorithm import DGA
from alg.Joint_Greedy_Algorithm import JGA
from alg.Random_Algorithm import RA
from alg.Random_Greedy_Algorithm import RGA
from env.evaluation import evaluation
from env.physical_scene import physical_scene
from plot.plot_eva import plot_eva
from plot.plot_time import plot_time

RAND_SEED = 2
# Set random seed for reproducibility
np.random.seed(RAND_SEED)

NUM_ITER = 100

# Define matrix types and names
matrix_types = ["Dct matrix", "Low correlation matrix", "Orthogonal matrix"]
matrix_type_names = ["Discrete Cosine Transform Matrix", "Low Correlation matrix", "Orthogonal Matrix"]

# Define parameters
N1, N2, N3 = 20, 30, 80
M1, M2 = 4, 10

M3_values = range(16, 44, 4)
x_labels = [str(M3) for M3 in M3_values]

# Initialize result and time arrays with an additional dimension for matrix types
results = np.zeros((len(matrix_types), 4, len(M3_values)))
execution_times = np.zeros((len(matrix_types), 4, len(M3_values)))



# Function to measure execution time
def measure_time_and_execute(func, *args):
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

for iteration in range(NUM_ITER):
    print(f"{iteration + 1} / {NUM_ITER}")

    for matrix_index, matrix_type in enumerate(matrix_types):
        # Generate the physical scene
        A, x, eta, sig = physical_scene(N1=N1, N2=N2, N3=N3, sig1=0.01, sig2=0.1, sig3=0.3, K=15, matrix=matrix_type)
        for i, M3 in enumerate(M3_values):
            # Measure time and evaluate each algorithm
            S_RGA, time_RGA = measure_time_and_execute(RGA, M1, M2, M3, N1, N2, N3, A, sig)
            S_JGA, time_JGA = measure_time_and_execute(JGA, M1, M2, M3, N1, N2, N3, A, sig)
            S_DGA, time_DGA = measure_time_and_execute(DGA, M1, M2, M3, N1, N2, N3, A)
            S_RA, time_RA = measure_time_and_execute(RA, M1, M2, M3, N1, N2, N3)

            # Evaluate the results
            results[matrix_index, 0, i] += evaluation(A, x, eta, sig, S_RGA)
            results[matrix_index, 1, i] += evaluation(A, x, eta, sig, S_JGA)
            results[matrix_index, 2, i] += evaluation(A, x, eta, sig, S_DGA)
            results[matrix_index, 3, i] += evaluation(A, x, eta, sig, S_RA)

            # Store execution times
            execution_times[matrix_index, 0, i] += time_JGA
            execution_times[matrix_index, 1, i] += time_RGA
            execution_times[matrix_index, 2, i] += time_DGA
            execution_times[matrix_index, 3, i] += time_RA

# Average results and times over iterations
results /= NUM_ITER
execution_times /= NUM_ITER

# Plot the evaluation results for each matrix type
plot_eva(results, x_labels, RAND_SEED, matrix_type_names)

# Plot the execution times (using the first matrix type as an example)
plot_time(execution_times[0], x_labels, RAND_SEED)