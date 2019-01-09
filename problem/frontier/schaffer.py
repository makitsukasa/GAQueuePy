import numpy as np

def schaffer(x):
	shifted = x * 200 - 100
	z = shifted[:-1] ** 2
	f = shifted[1:] ** 2
	return np.sum((z + f) ** 0.25 * (np.sin(50 * (z + f) ** 0.1) ** 2 + 1.0), dtype = np.float64)
