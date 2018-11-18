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
	parents = x[:n - 2]
	for i in range(3):
		rand = np.random.randint(len(x))
		parents.append(x[rand])
	return crossoverer.rex(parents)

def gaq3_op(x):
	try:
		clone = x[:]
		clone.sort(key = lambda i: -i.birth_year if i.birth_year is not None else 1)
		first = [clone.pop(0)]
		while clone[0].birth_year == first[0].birth_year:
			first.append(clone.pop(0))
		second = [clone.pop(0)]
		while clone[0].birth_year == second[0].birth_year:
			second.append(clone.pop(0))
		third = [clone.pop(0)]
		while clone[0].birth_year == third[0].birth_year:
			third.append(clone.pop(0))

		f = np.average([i.fitness for i in first])
		s = np.average([i.fitness for i in second])
		t = np.average([i.fitness for i in third])
		# if near local solution, search other place
		if (t - f) > (s - f):
			print("out", first[0].birth_year)
			pass
	except:
		pass
	clone = x[:]
	clone.sort(key = lambda i: i.fitness)
	return crossoverer.rex(clone[:n + 1])

n = 20
npop = 6 * n
npar = n + 1
nchi = 6 * n
step_count = 2000
problem = ackley

jggsys = JGGSystem(problem, n, npop, npar, nchi)
jggsys.step(step_count)
jggsys.calc_raw_fitness(problem)
best = jggsys.get_best_individual()
#print(best);
plot(step_count, jggsys.history, fmt = 'm-', label = 'JGG : {:.10f}'.format(best.fitness))

gaqsys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq_op)
gaqsys.step(step_count)
gaqsys.calc_raw_fitness(problem)
best = gaqsys.get_best_individual()
#print(best);
plot(step_count, gaqsys.history, fmt = 'b-', label = 'GAQ : {:.4f}'.format(best.fitness))

gaq2sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq2_op)
gaq2sys.step(step_count)
gaq2sys.calc_raw_fitness(problem)
best = gaq2sys.get_best_individual()
#print(best);
plot(step_count, gaq2sys.history, fmt = 'c-', label = 'GAQ2 : {:.4f}'.format(best.fitness))

gaq3sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq3_op)
gaq3sys.step(step_count)
gaq3sys.calc_raw_fitness(problem)
best = gaq3sys.get_best_individual()
print(best);
plot(step_count, gaq3sys.history, fmt = 'r-', label = 'GAQ3 : {:.4f}'.format(best.fitness))

plt.axis(xmin = 0, ymin = 0)
plt.title('{f}(D{d}), {s}'.format(f = problem.__name__, d = n, s = step_count))
plt.legend()
plt.show()
