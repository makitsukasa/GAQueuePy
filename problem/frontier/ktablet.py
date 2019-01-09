import numpy as np

def ktablet(x):
	shifted = x * 10.24 - 5.12
	k = len(x) // 4
	return np.sum(shifted[:k] ** 2) + np.sum((shifted[k:] * 100) ** 2)
