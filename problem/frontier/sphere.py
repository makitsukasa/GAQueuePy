import numpy as np

def sphere(x):
	shifted = x * 10.24 - 5.12
	return np.sum(shifted ** 2)
