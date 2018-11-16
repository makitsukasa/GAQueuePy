import numpy as np

def ackley(x):
	shifted = x * 65.536 - 32.768
	n = len(x)
	hoge = -0.2 * np.sqrt(np.sum(shifted ** 2) / n)
	piyo = np.sum(np.cos(2 * np.pi * shifted)) / n
	return 20 - 20 * np.exp(hoge) + np.e - np.exp(piyo)
