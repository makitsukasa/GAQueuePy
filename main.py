import numpy as np
from jggsystem import JGGSystem
from gaqsystem import GAQSystem
from individual import Individual
import crossoverer
from problem.frontier.sphere import sphere
from problem.frontier.ellipsoid import ellipsoid

def gaq_op(x):
	x.sort(key=lambda i: i.fitness)
	return crossoverer.rex(x[:n + 1])

def gaq2_op(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n]
	x.sort(key = lambda i: i.birth_year if i.birth_year is not None else -1)
	elders = x[:int(len(x) * 0.1)]
	elders.sort(key = lambda i: i.fitness)
	parents.extend(elders[:1])
	return crossoverer.rex(parents)

n = 20
problem = sphere
jggsys = JGGSystem(problem, n, 6 * n, n + 1, 6 * n)
gaqsys = GAQSystem(problem, 0, [Individual(n) for i in range(15 * n)], gaq_op)
gaq2sys = GAQSystem(problem, 0, [Individual(n) for i in range(15 * n)], gaq2_op)

jggsys.step(27200)
print(jggsys.get_best_individual());

gaqsys.step(27200)
print(gaqsys.get_best_individual());

gaq2sys.step(27200)
print(gaq2sys.get_best_individual());
