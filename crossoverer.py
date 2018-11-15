import numpy as np
from individual import Individual

def rex(parents, nchi = None):
	n = len(parents[0].gene)
	mu = len(parents)
	nchi = nchi if nchi else n + 1
	g = np.mean(np.array([parent.gene for parent in parents]), axis=0)
	children = [Individual(n) for i in range(nchi)]
	for child in children:
		epsilon = np.random.uniform(-np.sqrt(3 / mu), np.sqrt(3 / mu), mu)
		for i in range(n):
			child.gene[i] = g[i]
			for j in range(mu):
				child.gene[i] += (parents[j].gene[i] - g[i]) * epsilon[j]
	return children
