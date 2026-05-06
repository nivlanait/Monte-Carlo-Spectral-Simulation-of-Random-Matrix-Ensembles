import numpy as np

np.random.seed(281)

n = 100
A = np.random.normal(size=(n, n))
H = (A + A.T) / 2

np.savetxt("test_matrix.csv", H, delimiter=",")