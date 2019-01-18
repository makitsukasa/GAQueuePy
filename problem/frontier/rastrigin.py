import numpy as np

def rastrigin(x):
	# shifted = x * 10.24 - 5.12
	shifted = x * 5.12 - 2.56
	n = len(x)
	ret = 10 * n
	for i in range(n):
		ret += np.power(shifted[i], 2) - 10 * np.cos(2 * np.pi * shifted[i])
	return ret
