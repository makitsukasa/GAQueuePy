import numpy as np

MU = [
	[-1.0,  1.5, -2.0,  2.5],
	[ 0.0, -2.0,  3.0,  1.0],
	[-2.5, -2.0,  1.5,  3.5],
	[-2.0,  1.0, -1.0,  3.0],
]
SQ_SIGMA = [ 2.25, 4.0, 1.0, 4.0 ]
A = [ 3.1, 3.4, 4.1, 3.0 ]

def gmm(x):
	shifted = x * 10.24 - 5.12
	ans = 0.0
	for i in range(4):
		exponent = 0.0
		for j in range(len(shifted)):
			exponent -= ((shifted[j] - MU[i][j % 4]) ** 2) / (2 * SQ_SIGMA[i])
		ans += A[i] * np.exp(exponent)
	return -ans
