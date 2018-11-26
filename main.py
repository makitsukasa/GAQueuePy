import numpy as np
import matplotlib.pyplot as plt
from jggsystem import JGGSystem
from gaqsystem import GAQSystem
from individual import Individual
import crossoverer
from problem.frontier.sphere import sphere
from problem.frontier.ellipsoid import ellipsoid
from problem.frontier.ackley import ackley
from problem.frontier.rastrigin import rastrigin
from problem.gmm.gmm import gmm, rough_gmm_ave, rough_gmm_disc, init_rough_gmm
from plot import plot

def gaq_plain_op(x):
	x.sort(key=lambda i: i.fitness)
	return crossoverer.rex(x[:n + 1])

def gaq_always_random_op(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n - 1]
	for i in range(2):
		rand = np.random.randint(len(x))
		parents.append(x[rand])
	return crossoverer.rex(parents)

def gaq_rarely_random_op(x):
	if np.random.random() > 0.1:
		return gaq_always_random_op(x)
	else:
		return gaq_plain_op(x)

def gaq_fixed_range_op(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n - 1]
	clone = x[:]
	clone.sort(key = lambda i: -i.birth_year)
	clone = clone[:len(x) * 8 // 10]
	clone.sort(key = lambda i: i.fitness)
	parents.extend(clone[:2])
	return crossoverer.rex(parents)

def gaq_random_range_op(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n - 1]
	clone = x[:]
	clone.sort(key = lambda i: -i.birth_year)
	clone = clone[:len(x) * np.random.randint(1, 100) // 100]
	clone.sort(key = lambda i: i.fitness)
	parents.extend(clone[:2])
	return crossoverer.rex(parents)

n = 10
npop = 6 * n
npar = 4
nchi = 16
step_count = 10000
loop_count = 100
problem = rough_gmm_ave
raw_problem = gmm
gaqsystem_opt_list = [
	["plain", "m"],
	["always_random", "b"],
	["rarely_random", "navy"],
	["fixed_range", "c"],
	["random_range", "g"],
]
best_list = {"jgg" : 0}
for opt in gaqsystem_opt_list:
	name, color = opt
	best_list[name] = 0

for _ in range(loop_count):
	randseed = np.random.randint(0x7fffffff)

	init_rough_gmm()
	np.random.seed(randseed)
	jggsys = JGGSystem(problem, n, npop, npar, nchi)
	jggsys.step(step_count)
	jggsys.calc_raw_fitness(raw_problem)
	best = jggsys.get_best_individual()
	# print(best);
	best_list["jgg"] += best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, jggsys.history, color = 'r', label = 'JGG : {:.10f}'.format(best.raw_fitness))

	for opt in gaqsystem_opt_list:
		name, color = opt
		exec("op = gaq_{}_op".format(name))
		init_rough_gmm()
		np.random.seed(randseed)
		gaq_sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], op)
		gaq_sys.step(step_count)
		gaq_sys.calc_raw_fitness(raw_problem)
		best = gaq_sys.get_best_individual()
		# print(best);
		best_list[name] += best.raw_fitness / loop_count
		if loop_count == 1:
			plot(step_count, gaq_sys.history, color = color, label = 'GAQ_{} : {:.10f}'.format(name, best.raw_fitness))

	if loop_count == 1:
		# plt.axis(xmin = 0, ymin = 0)
		plt.title('{f}(D{d}), {npop},{npar},{s}'.format(f = problem.__name__, d = n, npop = npop, npar = npar, s = step_count))
		plt.legend()
		plt.show()

for key, ave in best_list.items():
	print(key, ave)
