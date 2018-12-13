import random
import array

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from problem.gmm.gmm import gmm

def gmm_wrapped(x):
	return gmm(numpy.array(x)),

# Problem dimension
NDIM = 10

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", gmm_wrapped)

def main():
	# Differential evolution parameters
	CR = 0.7
	F = 0.3
	MU = 20
	NGEN = 1000

	pop = toolbox.population(n=MU);
	hof = tools.HallOfFame(1)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"

	# Evaluate the individuals
	fitnesses = toolbox.map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	record = stats.compile(pop)
	logbook.record(gen=0, evals=len(pop), **record)
	# print(logbook.stream)

	for g in range(1, NGEN):
		for k, agent in enumerate(pop):
			a,b,c = toolbox.select(pop)
			y = toolbox.clone(agent)
			index = random.randrange(NDIM)
			for i, value in enumerate(agent):
				if i == index or random.random() < CR:
					y[i] = a[i] + F*(b[i]-c[i])
			y.fitness.values = toolbox.evaluate(y)
			if abs(y.fitness.values[0] - agent.fitness.values[0]) < 0.01 and random.random() < 0.5:
				pop[k] = y
			elif y.fitness > agent.fitness:
				pop[k] = y
		hof.update(pop)
		record = stats.compile(pop)
		logbook.record(gen=g, evals=len(pop), **record)
		# print(logbook.stream)

	# print("Best individual is ", hof[0], hof[0].fitness.values[0])
	return hof[0].fitness.values[0]

if __name__ == "__main__":
	result = []
	for _ in range(100):
		result.append(main())

	print(numpy.average(result))
	# main()
