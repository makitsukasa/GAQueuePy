import numpy as np

def rastrigin(x):
	shifted = x * 10.24 - 5.12 - 1
	hoge = shifted ** 2 - 10 * np.cos(2 * np.pi * shifted) + 10
	return np.sum(hoge)
