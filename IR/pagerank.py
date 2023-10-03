import numpy as np

MAX_ITERATIONS = 1000
CONVERGENCE_THREASHOLD = 0.01

# input
input_mat = [
  [1, 1, 1],
  [1, 0, 1],
  [0, 1, 1],
]
alpha = 0.5

n = len(input_mat)
A = np.matrix(input_mat)
col_sums = np.sum(A, axis=1)
A = A/col_sums
A = A.transpose()
A = (1-alpha) * A
A = A + (alpha/n)

p = np.matrix([1/n] * n).reshape(-1, 1)

for i in range(MAX_ITERATIONS):
  print(f'p{i}:', p, sep='\n')
  p_new = np.matmul(A, p)

  if max(abs(p_new - p)) < CONVERGENCE_THREASHOLD:
    break

  p = p_new.copy()
  