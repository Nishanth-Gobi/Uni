# Hyperlink Induced Topic Search (HITS)

import numpy as np

MAX_ITERATIONS = 1000
CONVERGENCE_THREASHOLD = 0.01

# Input
input_mat = [
  [0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [1, 0, 1, 1, 1, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 1, 0, 0]
]

A = np.matrix(input_mat)
A_T = A.transpose()

a = np.matrix([1] * len(input_mat)).reshape(-1, 1)
a = a/sum(a) # normalising

h = np.matrix([1] * len(input_mat)).reshape(-1, 1)
h = h/sum(h) # normalising

for i in range(MAX_ITERATIONS):
  print(f'a{i}', a, sep='\n')
  print(f'h{i}', h, sep='\n', end='\n')

  h_new = np.matmul(A, a)
  h_new = h_new/sum(h_new)

  a_new = np.matmul(A_T, h)
  a_new = a_new/sum(a_new)

  max_h_diff = max(abs(h_new-h))
  max_a_diff = max(abs(a_new-a))

  if max_a_diff < CONVERGENCE_THREASHOLD and max_h_diff < CONVERGENCE_THREASHOLD:
    break

  h = h_new.copy()
  a = a_new.copy()
  