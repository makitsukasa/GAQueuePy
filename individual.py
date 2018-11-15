import numpy as np

class Individual:
	def __init__(self, n):
		# initialized by random values
		if isinstance(n, int):
			self.n = n
			self.gene = np.random.uniform(0, 1, n)

		# initialized by list
		else:
			self.n = len(n)
			self.gene = n

		self.fitness = None
		self.raw_fitness = None
		self.birth_year = None

	def __str__(self):
		return "{r}/{f}({b}) {g}".format(
			r = self.raw_fitness,
			f = self.fitness,
			b = self.birth_year,
			g = self.gene
		)
