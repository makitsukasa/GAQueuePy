import numpy as np
from individual import Individual
import crossoverer
from jggsystem import JGGSystem
from gaqsystem import GAQSystem

def is_stucked(x, t = 1e-7):
	clone = x[:]
	clone.sort(key = lambda i: i.birth_year)
	oldest_birth_year = clone[0].birth_year
	clone.sort(key = lambda i: -i.birth_year)
	most_recent_birth_year = clone[0].birth_year
	for i in clone:
		if i.birth_year == most_recent_birth_year:
			continue
		second_recent_birth_year = i.birth_year
		break
	if not "second_recent_birth_year" in locals():
		return False
	if oldest_birth_year == second_recent_birth_year:
		return False

	init_pop = [i for i in x if i.birth_year == oldest_birth_year]
	most_recent_pop = [i for i in x if i.birth_year == most_recent_birth_year]
	second_recent_pop = [i for i in x if i.birth_year == second_recent_birth_year]

	init_fitness = np.average([i.fitness for i in init_pop])
	most_recent_fitness = np.average([i.fitness for i in most_recent_pop])
	second_recent_fitness = np.average([i.fitness for i in second_recent_pop])

	diff_init_rec = init_fitness - most_recent_fitness
	diff_rec_rec = second_recent_fitness - most_recent_fitness
	if diff_init_rec != 0 and abs(diff_rec_rec / diff_init_rec) >= t:
		return False

	return True

def gaq_op_plain_origopt(x):
	x.sort(key=lambda i: i.fitness)
	n = x[0].n
	return crossoverer.rex(x[:n + 1])

class SwapSystem(object):
	def __init__(self, problem, raw_problem, t, n, npop, npar, nchi):
		self.t = t
		self.n = n
		self.problem = problem
		self.npop = npop
		self.npar = npar
		self.nchi = nchi
		self.history = []
		self.is_gaq_active = True

		self.gaq_sys = GAQSystem(
			problem,
			raw_problem,
			0,
			[Individual(n) for i in range(npop)],
			gaq_op_plain_origopt
		)
		self.jgg_sys = JGGSystem(problem, raw_problem, n, npop, npar, nchi)

	def get_active_system(self):
		if self.is_gaq_active:
			return self.gaq_sys
		else:
			return self.jgg_sys

	def switch_to_gaq(self, gaq_sys):
		return is_stucked(gaq_sys.history, self.t)

	def switch_to_jgg(self, jgg_sys):
		return is_stucked(jgg_sys.history, self.t)

	def choose_population_to_jgg(self, gaq_sys):
		np.random.shuffle(gaq_sys.history)
		return gaq_sys.history[:self.npop]

	def switch_active_system(self):
		if self.is_gaq_active:
			if not self.switch_to_jgg(self.gaq_sys):
				return
			# print("GAQ->JGG", self.gaq_sys.age)
			self.jgg_sys.history = self.gaq_sys.history[:]
			self.jgg_sys.population = self.choose_population_to_jgg(self.gaq_sys)
			self.jgg_sys.age = self.gaq_sys.age
			self.jgg_sys.generate_first_children()
			self.is_gaq_active = False
		else:
			if not self.switch_to_gaq(self.jgg_sys.history):
				return
			# print("JGG->GAQ", self.jgg_sys.age)
			self.gaq_sys.history = self.jgg_sys.history
			self.gaq_sys.age = self.jgg_sys.age
			self.is_gaq_active = True

	def step(self, count = 1):
		for _ in range(count):
			self.get_active_system().step()
			self.switch_active_system()

	def until_goal(self, goal = 10e-7, max_count = 200000):
		for _ in range(max_count):
			active_system = self.get_active_system()
			if not hasattr(active_system, "until_goal"):
				active_system.step()
				self.switch_active_system()
			else:
				return active_system.until_goal(goal, max_count - self.gaq_sys.age)

		return False

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
