import numpy as np
from individual import Individual
import crossoverer

class GAQSystem(object):
	"""
	overview diagram here
	each arrow shows flow of individual data
	bold arrow shows that 2 individual flows there at once

	(evaluate) <--(popQueue)-- [queue] <=====
	 |                                      ||
	(addHistory)                            ||
	 |                                      ||
	 V                                      ||
	[history] ==(  s u p p l y Q u e u e  )==
	             * * * * * * * * * * * * *
	             * select 2 from history *
	             * crossover selected 2  *
	             * mutate each new indiv *
	             * push new 2 to queue   *
	             * * * * * * * * * * * * *

	these have some options to choose from
			- size of gene
			- evaluate  function
			- select    function
			- crossover function
			- mutate    function
	"""

	def __init__(self, problem, minQueueSize, firstGeneration, op):
		self.problem = problem
		self.minQueueSize = minQueueSize
		self.queue = firstGeneration
		self.op = op
		self.history = []
		self.age = 0

	def evaluate(self, indiv):
		indiv.fitness = self.problem(indiv.gene)
		return indiv

	def step(self, count = 1):
		for i in range(count):
			self.age += 1
			evaluated = self.evaluate(self.queue.pop(-1))
			self.history.append(evaluated)
			if len(self.queue) <= self.minQueueSize:
				new_generation = self.op(self.history)
				for i in new_generation:
					i.birth_year = self.age
				self.queue.extend(new_generation)

	def calc_raw_fitness(problem):
		for i in history:
			i.raw_fitness = problem(i.gene)

if __name__ == '__main__':

	def sphere(x):
		shifted = x * 10.24 - 5.12
		return np.sum(shifted ** 2)

	def op(x):
		x.sort(key=lambda i: i.fitness)
		return crossoverer.rex(x[:npar])

	n = 20
	nfirst = n + 1
	npar = n + 1
	system = GAQSystem(
		sphere,
		0,
		[Individual(n) for i in range(nfirst)],
		op
	)

	system.step(800)

	system.history.sort(key = lambda i: i.fitness)
	print(system.history[0].fitness)
	# 10e-7 or lower

