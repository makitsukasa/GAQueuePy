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

def gaq_op(x):
	x.sort(key=lambda i: i.fitness)
	return crossoverer.rex(x[:n + 1])

def gaq2_op(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n]
	rand = np.random.randint(len(x))
	parents.append(x[rand])
	return crossoverer.rex(parents)

n = 20
npop = 22 * n
npar = n + 1
nchi = 8 * n
step_count = 220000
problem = rastrigin
jggsys = JGGSystem(problem, n, npop, npar, nchi)
gaqsys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq_op)
gaq2sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq2_op)

jggsys.step(step_count)
jggsys.calc_raw_fitness(problem)
best = jggsys.get_best_individual()
print(best);
plot(step_count, jggsys.history, fmt = 'm-', label = 'JGG : {:.10f}'.format(best.fitness))

gaqsys.step(step_count)
gaqsys.calc_raw_fitness(problem)
best = gaqsys.get_best_individual()
print(best);
plot(step_count, gaqsys.history, fmt = 'b-', label = 'GAQ : {:.4f}'.format(best.fitness))

gaq2sys.step(step_count)
gaq2sys.calc_raw_fitness(problem)
best = gaq2sys.get_best_individual()
print(best);
plot(step_count, gaq2sys.history, fmt = 'c-', label = 'GAQ2 : {:.4f}'.format(best.fitness))

plt.axis(xmin = 0, ymin = 0)
plt.title('{f}(D{d}), {s}'.format(f = problem.__name__, d = n, s = step_count))
plt.legend()
plt.show()
