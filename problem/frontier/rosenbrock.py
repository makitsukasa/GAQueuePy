import numpy as np

def rosenbrock(x):
	shifted = x * 4.096 - 2.048
	return np.sum(100 * (shifted[0] - shifted[1:] ** 2) ** 2 + (1 - shifted[1:]) ** 2)
