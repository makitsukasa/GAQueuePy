import numpy as np
from individual import Individual
import crossoverer
from jggsystem import JGGSystem
from gaqsystem import GAQSystem

fitness_history = []

def is_stucked(x):
	global fitness_history
	delay = 5

	clone = x[:]
	clone.sort(key = lambda i: -i.birth_year)
	most_recent_birth_year = clone[0].birth_year
	most_recent_pop = [i for i in x if i.birth_year == most_recent_birth_year]
	fitness_history.append(np.average([i.fitness for i in most_recent_pop]))

	if len(fitness_history) < delay + 1:
		return False

	for n in range(delay):
		diff_init_rec = fitness_history[0] - fitness_history[-1 - n]
		diff_rec_rec = fitness_history[-2 - n] - fitness_history[-1 - n]
		if diff_init_rec != 0 and abs(diff_rec_rec / diff_init_rec) >= 0.000001:
			return False

	return True

def gaq_op_plain_origopt(x):
	x.sort(key=lambda i: i.fitness)
	n = x[0].n
	return crossoverer.rex(x[:n + 1])

class SwapSystem(object):
	def __init__(self, problem, n, npop, npar, nchi):
		global fitness_history
		fitness_history.clear()
		self.n = n
		self.problem = problem
		self.npop = npop
		self.npar = npar
		self.nchi = nchi
		self.history = []
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

	def switch_to_gaq(self, history):
		return is_stucked(history)

	def switch_active_system(self):
		global fitness_history
		if self.is_gaq_active:
			if is_stucked(self.gaq_sys.history):
				print("JGG->GAQ", self.gaq_sys.age)
				self.jgg_sys.history = self.gaq_sys.history
				np.random.shuffle(self.gaq_sys.history)
				self.jgg_sys.population = self.gaq_sys.history[:self.npop]
				self.jgg_sys.age = self.gaq_sys.age
				self.is_gaq_active = False
				fitness_history.clear()
		else:
			if self.switch_to_gaq(self.jgg_sys.history):
				print("GAQ->JGG", self.jgg_sys.age)
				self.gaq_sys.history = self.jgg_sys.history
				self.gaq_sys.age = self.jgg_sys.age
				self.is_gaq_active = True
				fitness_history.clear()

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
	system.switch_to_gaq = lambda x: False
	system.step(27200)
	system.calc_raw_fitness(sphere)
	print(system.get_best_individual())
