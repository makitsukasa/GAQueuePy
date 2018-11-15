from individual import Individual
import numpy as np

class JGGSystem(object):
	def __init__(self, func, n, npop, npar, nchi):
		self.n = n
		self.func = func
		self.children = []
		self.npop = npop
		self.npar = npar
		self.nchi = nchi

		self.population = [Individual(self.n) for i in range(npop)]

		parents = self.select_parents()
		self.children_before_eval = self.rex(parents)
		self.children_after_eval = []

	def select_parents(self):
		np.random.shuffle(self.population)
		self.parents = self.population[:self.npar]
		self.population = self.population[self.npar:]
		return self.parents

	def rex(self, parents):
		mu = len(parents)
		g = np.mean(np.array([parent.gene for parent in parents]), axis=0)
		self.children_before_eval = [Individual(self.n) for i in range(self.nchi)]
		for child in self.children_before_eval:
			epsilon = np.random.uniform(-np.sqrt(3 / (mu)), np.sqrt(3 / (mu)), mu)
			for i in range(self.n):
				child.gene[i] = g[i]
				for j in range(mu):
					child.gene[i] += (parents[j].gene[i] - g[i]) * epsilon[j]
		return self.children_before_eval

	def survival_selection(self, evaluated):
		evaluated.sort(key=lambda child: child.f)
		ret = evaluated[:self.npar]
		return ret

	def evaluate(self, indiv):
		indiv.f = self.func(indiv.gene)
		self.children.append(indiv)
		return indiv

	def get_best_evaluation_value(self):
		self.population.sort(key=lambda s: s.f)
		return self.population[0].f, self.population[0].gene

	def step(self, count = 1):
		for i in range(count):
			evaluated = self.evaluate(self.children_before_eval.pop(-1))
			self.children_after_eval.append(evaluated)
			if len(self.children_before_eval) == 0:
				new_generation = self.survival_selection(self.children_after_eval)
				self.population.extend(new_generation)
				self.children_after_eval.clear()
				taken = self.select_parents()
				self.children_before_eval = self.rex(taken)

if __name__ == '__main__':

	def sphere(x):
		shifted = x * 10.24 - 5.12
		return np.sum(shifted ** 2)

	n = 20
	system = JGGSystem(sphere, n, 6 * n, n + 1, 6 * n)

	system.step(30300)

	print(system.get_best_evaluation_value());
	# 10e-7 or lower
