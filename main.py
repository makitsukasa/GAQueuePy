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
	if np.random.random() > 0.5:
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

n = 20
npop = 6 * n
npar = n + 1
nchi = 6 * n
step_count = 27200
problem = sphere
gaqsystem_opt_list = [
	["plain", "m"],
	["always_random", "b"],
	["rarely_random", "navy"],
	["fixed_range", "c"],
	["random_range", "g"],
]
randseed = np.random.randint(0x7fffffff)

np.random.seed(randseed)
jggsys = JGGSystem(problem, n, npop, npar, nchi)
jggsys.step(step_count)
jggsys.calc_raw_fitness(problem)
best = jggsys.get_best_individual()
print(best);
plot(step_count, jggsys.history, color = 'r', label = 'JGG : {:.10f}'.format(best.fitness))

for opt in gaqsystem_opt_list:
	name, color = opt
	exec("op = gaq_{}_op".format(name))
	np.random.seed(randseed)
	gaq_sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], op)
	gaq_sys.step(step_count)
	gaq_sys.calc_raw_fitness(problem)
	best = gaq_sys.get_best_individual()
	print(best);
	plot(step_count, gaq_sys.history, color = color, label = 'GAQ_{} : {:.10f}'.format(name, best.fitness))

plt.axis(xmin = 0, ymin = 0)
plt.title('{f}(D{d}), {npop},{npar},{s}'.format(f = problem.__name__, d = n, npop = npop, npar = npar, s = step_count))
plt.legend()
plt.show()
