import numpy as np
from enum import Enum, auto

class State(Enum):
	NONE = auto()
	SEARCHING = auto()
	NO_LONGER_SEARCH = auto()
	USED_IN_GAQ = auto()

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
		self.birth_year = 0
		self.state = State.NONE

	def __str__(self):
		return "{r}/{f}({b}) {g}".format(
			r = 'None' if self.raw_fitness is None else '{:.4}'.format(self.raw_fitness),
			f = 'None' if self.fitness     is None else '{:.4}'.format(self.fitness),
			b = 'None' if self.birth_year  is None else self.birth_year,
			g = ','.join(['{:.4}'.format(g) for g in self.gene])
		)
