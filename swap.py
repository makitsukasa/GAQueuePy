import numpy as np
import matplotlib.pyplot as plt
from jggsystem import JGGSystem
from gaqsystem import GAQSystem
from sggsystem import SGGSystem
from individual import Individual, State
import crossoverer
from problem.frontier.sphere import sphere
from problem.frontier.ellipsoid import ellipsoid
from problem.frontier.ackley import ackley
from problem.frontier.rastrigin import rastrigin
from problem.gmm.gmm import gmm, rough_gmm_ave, rough_gmm_weighted_ave, init_rough_gmm
from plot import plot
import warnings

warnings.simplefilter("error", RuntimeWarning)

def gaq_op_plain_origopt(x):
	x.sort(key=lambda i: i.fitness)
	return crossoverer.rex(x[:n + 1])

def gaq_op_plain_jggopt(x):
	x.sort(key=lambda i: i.fitness)
	return crossoverer.rex(x[:npar], nchi)

def gaq_op_always_random_origopt(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n - 1]
	for i in range(2):
		rand = np.random.randint(len(x))
		parents.append(x[rand])
	return crossoverer.rex(parents)

def gaq_op_always_random_jggopt(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:npar - 2]
	for i in range(2):
		rand = np.random.randint(len(x))
		parents.append(x[rand])
	return crossoverer.rex(parents, nchi)

def gaq_op_rarely_random(x):
	if np.random.random() > 0.1:
		return gaq_op_always_random_origopt(x)
	else:
		return gaq_op_plain_origopt(x)

def gaq_op_fixed_range(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n - 1]
	clone = x[:]
	clone.sort(key = lambda i: -i.birth_year)
	clone = clone[:len(x) * 8 // 10]
	clone.sort(key = lambda i: i.fitness)
	parents.extend(clone[:2])
	return crossoverer.rex(parents)

def gaq_op_random_range(x):
	x.sort(key = lambda i: i.fitness)
	parents = x[:n - 1]
	clone = x[:]
	clone.sort(key = lambda i: -i.birth_year)
	clone = clone[:len(x) * np.random.randint(1, 100) // 100]
	clone.sort(key = lambda i: i.fitness)
	parents.extend(clone[:2])
	return crossoverer.rex(parents)

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

def gaq_op_plain_origopt_gradient(x):
	trimmed = [i for i in x if i.state != State.NO_LONGER_SEARCH]
	if is_stucked(trimmed):
		for i in x:
			if i.state == State.SEARCHING:
				i.state = State.NO_LONGER_SEARCH
		trimmed = [i for i in x if i.state != State.NO_LONGER_SEARCH]
		trimmed.sort(key=lambda i: i.fitness)
		for i in trimmed[:2]:
			i.state = State.NO_LONGER_SEARCH

	trimmed.sort(key=lambda i: i.fitness)
	ret = crossoverer.rex(trimmed[:n + 1])
	for i in ret:
		i.state = State.SEARCHING
	return ret

def gaq_op_plain_jggopt_gradient_elitesurvive(x):
	trimmed = [i for i in x if i.state != State.NO_LONGER_SEARCH]
	if is_stucked(trimmed):
		searching = [i for i in x if i.state == State.SEARCHING]
		searching.sort(key = lambda i: i.fitness)
		for i in searching[npar:]:
			i.state = State.NO_LONGER_SEARCH
		for i in searching[:npar]:
			i.state = State.NONE
		trimmed = [i for i in x if i.state != State.NO_LONGER_SEARCH]
		np.random.shuffle(trimmed)
		ret = crossoverer.rex(trimmed[:npar], nchi)

	else:
		trimmed.sort(key=lambda i: i.fitness)
		ret = crossoverer.rex(trimmed[:npar], nchi)

	for i in ret:
		i.state = State.SEARCHING
	return ret

def gaq_op_pseudo_jgg(x):
	np.random.shuffle(x)
	children = crossoverer.rex(x[:npar], nchi)

def init():
	init_rough_gmm()
	max_gradient = 0.0

def rough_gmm_ave_50(x):
	return rough_gmm_ave(x, 0.5)
def rough_gmm_ave_100(x):
	return rough_gmm_ave(x, 1.0)
def rough_gmm_ave_200(x):
	return rough_gmm_ave(x, 2.0)
def rough_gmm_weighted_ave_25(x):
	return rough_gmm_weighted_ave(x, 0.25)
def rough_gmm_weighted_ave_50(x):
	return rough_gmm_weighted_ave(x, 0.5)
def rough_gmm_weighted_ave_75(x):
	return rough_gmm_weighted_ave(x, 0.75)

n = 20
npop = 6 * n
npar = n + 1
nchi = 6 * n
swap_count = 25000
step_count = 27200
loop_count = 1
problem = sphere
raw_problem = sphere
title = '{f}(D{d}), pop{npop},par{npar},chi{nchi},step{s},loop{l}'.format(
	f = problem.__name__, d = n, npop = npop, npar = npar, nchi = nchi, s = step_count, l = loop_count)
best_list = {"jgg" : 0, "gaq" : 0, "jgg->gaq" : 0, "gaq->jgg" : 0}
print(title)

for _ in range(loop_count):
	randseed = np.random.randint(0x7fffffff)

	np.random.seed(randseed)
	jgg_sys = JGGSystem(problem, n, npop, npar, nchi)
	jgg_sys.step(step_count)
	jgg_sys.calc_raw_fitness(raw_problem)
	best = jgg_sys.get_best_individual()
	best_list["jgg"] += best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, jgg_sys.history, color = 'r', label = 'JGG : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	gaq_sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq_op_plain_origopt)
	gaq_sys.step(step_count)
	gaq_sys.calc_raw_fitness(raw_problem)
	best = gaq_sys.get_best_individual()
	best_list["gaq"] += best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, gaq_sys.history, color = 'b', label = 'GAQ : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	gaq_jgg_sys = JGGSystem(problem, n, npop, npar, nchi)
	gaq_jgg_sys.step(swap_count)

	op = gaq_op_plain_origopt
	np.random.seed(randseed)
	jgg_gaq_sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], op)
	jgg_gaq_sys.step(swap_count)

	his_JGG = gaq_jgg_sys.history[:]
	his_GAQ = jgg_gaq_sys.history[:]

	gaq_jgg_sys.history = his_GAQ
	gaq_jgg_sys.population = his_GAQ[:npop]
	jgg_gaq_sys.history = his_JGG

	gaq_jgg_sys.step(step_count - swap_count)
	gaq_jgg_sys.calc_raw_fitness(raw_problem)
	best = gaq_jgg_sys.get_best_individual()
	best_list["gaq->jgg"] += best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, gaq_jgg_sys.history, color = 'g', label = 'GAQ->JGG : {:.10f}'.format(best.raw_fitness))

	jgg_gaq_sys.step(step_count - swap_count)
	jgg_gaq_sys.calc_raw_fitness(raw_problem)
	best = jgg_gaq_sys.get_best_individual()
	best_list["jgg->gaq"] += best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, jgg_gaq_sys.history, color = 'm', label = 'JGG->GAQ : {:.10f}'.format(best.raw_fitness))

	if loop_count == 1:
		# plt.axis(xmin = 0, ymin = 0)
		plt.title(title)
		plt.legend()
		plt.show()

for key, ave in best_list.items():
	print(key, ave)
