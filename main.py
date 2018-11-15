import numpy as np
from jggsystem import JGGSystem

def sphere(x):
	shifted = x * 10.24 - 5.12
	return np.sum(shifted ** 2)

n = 20
system = JGGSystem(sphere, n, 6 * n, n + 1, 6 * n)

system.step(30300)

print(system.get_best_evaluation_value());
