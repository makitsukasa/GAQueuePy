import numpy as np

def ellipsoid(x):
	shifted = x * 10.24 - 5.12
	n = len(x)
	ret = 0
	for i in range(n):
		ret += 10.0 ** (6.0 * i / (n - 1)) * shifted[i] ** 2
	return ret
