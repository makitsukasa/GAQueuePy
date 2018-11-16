import numpy as np
from individual import Individual
import crossoverer

class JGGSystem(object):
	def __init__(self, problem, n, npop, npar, nchi):
		self.n = n
		self.problem = problem
		self.children = []
		self.npop = npop
		self.npar = npar
		self.nchi = nchi
		self.history = []
		self.age = 0

		self.population = [Individual(self.n) for i in range(npop)]

		parents = self.select_parents()
		self.children_before_eval = crossoverer.rex(parents, self.nchi)
		self.children_after_eval = []

	def select_parents(self):
		np.random.shuffle(self.population)
		self.parents = self.population[:self.npar]
		self.population = self.population[self.npar:]
		return self.parents

	def survival_selection(self, evaluated):
		evaluated.sort(key=lambda child: child.fitness)
		ret = evaluated[:self.npar]
		return ret

	def evaluate(self, indiv):
		indiv.fitness = self.problem(indiv.gene)
		self.children.append(indiv)
		return indiv

	def get_best_individual(self):
		self.history.sort(key=lambda s: s.raw_fitness)
		return self.history[0]

	def step(self, count = 1):
		for i in range(count):
			self.age += 1
			evaluated = self.evaluate(self.children_before_eval.pop(-1))
			self.children_after_eval.append(evaluated)
			self.history.append(evaluated)
			if len(self.children_before_eval) == 0:
				new_generation = self.survival_selection(self.children_after_eval)
				self.population.extend(new_generation)
				self.children_after_eval.clear()
				taken = self.select_parents()
				self.children_before_eval = crossoverer.rex(taken, self.nchi)
				for i in self.children_before_eval:
					i.birth_year = self.age

	def calc_raw_fitness(self, problem):
		for i in self.history:
			i.raw_fitness = problem(i.gene)

if __name__ == '__main__':

	def sphere(x):
		shifted = x * 10.24 - 5.12
		return np.sum(shifted ** 2)

	n = 20
	system = JGGSystem(sphere, n, 6 * n, n + 1, 6 * n)

	system.step(30300)

	print(system.get_best_evaluation_value());
	# 10e-7 or lower
