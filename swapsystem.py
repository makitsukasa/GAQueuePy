import numpy as np
from individual import Individual
import crossoverer
from jggsystem import JGGSystem
from gaqsystem import GAQSystem

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

def gaq_op_plain_origopt(x):
	x.sort(key=lambda i: i.fitness)
	n = x[0].n
	return crossoverer.rex(x[:n + 1])

class SwapSystem(object):
	def __init__(self, problem, n, npop, npar, nchi):
		self.n = n
		self.problem = problem
		self.npop = npop
		self.npar = npar
		self.nchi = nchi
		self.history = []
		self.age = 0
		self.is_gaq_active = True

		self.jgg_sys = JGGSystem(problem, n, npop, npar, nchi)
		self.gaq_sys = GAQSystem(
			problem,
			0,
			[Individual(n) for i in range(npop)],
			gaq_op_plain_origopt
		)

	def get_active_system(self):
		if self.is_gaq_active:
			return self.gaq_sys
		else:
			return self.jgg_sys

	def switch_active_system(self):
		if self.is_gaq_active:
			if is_stucked(self.gaq_sys.history):
				self.jgg_sys.history = self.gaq_sys.history
				np.random.shuffle(self.gaq_sys.history)
				self.jgg_sys.population = self.gaq_sys.history[:self.npop]
				self.jgg_sys.age = self.gaq_sys.age
				self.is_gaq_active = False
		else:
			if False:
				self.gaq_sys.history = self.jgg_sys.history
				self.gaq_sys.age = self.jgg_sys.age
				self.is_gaq_active = True

	def step(self, count = 1):
		for _ in range(count):
			self.get_active_system().step()
			self.switch_active_system()

	def calc_raw_fitness(self, problem):
		self.get_active_system().calc_raw_fitness(problem)

	def get_best_individual(self):
		self.get_active_system().history.sort(key=lambda s: s.raw_fitness)
		return self.get_active_system().history[0]

if __name__ == '__main__':
	def sphere(x):
		shifted = x * 10.24 - 5.12
		return np.sum(shifted ** 2)

	n = 20
	system = SwapSystem(sphere, n, 6 * n, n + 1, 6 * n)
	system.step(27200)
	system.calc_raw_fitness(sphere)
	print(system.get_best_individual())
