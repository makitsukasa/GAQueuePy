import numpy as np
from individual import Individual
import crossoverer

def is_stucked(x):
	initial = [i for i in x if i.birth_year == 0]
	x.sort(key = lambda i: -i.birth_year)
	most_recent_birth_year = x[0].birth_year
	most_recent = [i for i in x if i.birth_year == most_recent_birth_year]
	left = [i for i in x if i not in most_recent]

	if len(initial) == 0 or len(left) == 0:
		return False

	second_recent_birth_year = left[0].birth_year
	second_recent = [i for i in left if i.birth_year == second_recent_birth_year]

	initial_fitness = np.average([i.fitness for i in initial])
	most_recent_fitness = np.average([i.fitness for i in most_recent])
	second_recent_fitness = np.average([i.fitness for i in second_recent])

	diff_init_mostrecent = initial_fitness - most_recent_fitness
	diff_mostrecent_secondrecent = second_recent_fitness - most_recent_fitness

	if diff_init_mostrecent == 0 or abs(diff_mostrecent_secondrecent / diff_init_mostrecent) < 0.000001:
		# print("stucked", most_recent_birth_year, diff_mostrecent_secondrecent, diff_init_mostrecent)
		return True
	else:
		return False

class SGGSystem(object):

	def __init__(self, problem, n, npop, npar, max_inner_age = None):
		self.n = n
		self.problem = problem
		self.children = []
		self.npop = npop
		self.npar = npar
		self.history = []
		self.age = 0
		self.inner_age = 0
		self.max_inner_age = max_inner_age

		self.outer_population = [Individual(self.n) for i in range(npop)]

		self.inner_population = []
		self.children_before_eval = self.select_parents()
		self.children_after_eval = []

	def select_parents(self):
		np.random.shuffle(self.outer_population)
		parents = self.outer_population[:self.npar]
		self.outer_population = self.outer_population[self.npar:]
		return parents

	def evaluate(self):
		indiv = self.children_before_eval.pop(-1)
		indiv.fitness = self.problem(indiv.gene)
		self.children.append(indiv)
		self.children_after_eval.append(indiv)
		self.history.append(indiv)
		return indiv

	def can_go_next_generation(self):
		if self.max_inner_age is None:
			return is_stucked(self.history)
		else:
			return self.inner_age >= self.max_inner_age

	def step(self, count = 1):
		for i in range(count):
			self.age += 1
			self.evaluate()
			if len(self.children_before_eval) == 0:
				self.inner_age += 1
				self.inner_population.extend(self.children_after_eval)
				self.children_after_eval.clear()
				if not self.can_go_next_generation():
					self.inner_population.sort(key=lambda i: i.fitness)
					self.children_before_eval = crossoverer.rex(self.inner_population[:self.npar])
				else:
					self.inner_population.sort(key=lambda i: i.fitness)
					self.outer_population.extend(self.inner_population[:self.npar])
					self.inner_population.clear()
					self.children_before_eval = self.select_parents()
					self.inner_age = 0
				for i in self.children_before_eval:
					i.birth_year = self.age

	def calc_raw_fitness(self, problem):
		for i in self.history:
			i.raw_fitness = problem(i.gene)

	def get_best_individual(self):
		self.history.sort(key=lambda s: s.raw_fitness)
		return self.history[0]

if __name__ == '__main__':

	def sphere(x):
		shifted = x * 10.24 - 5.12
		return np.sum(shifted ** 2)

	n = 20
	system = SGGSystem(sphere, n, 6 * n, n + 1)
	system.step(27200)
	system.calc_raw_fitness(sphere)
	print(system.get_best_individual())
