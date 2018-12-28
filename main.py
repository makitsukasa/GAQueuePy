import numpy as np
import matplotlib.pyplot as plt
from jggsystem import JGGSystem
from gaqsystem import GAQSystem
from sggsystem import SGGSystem
from swapsystem import SwapSystem
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
	x.sort(key = lambda i : i.fitness)
	return crossoverer.rex(x[:n + 1])

def choose_population_throw_gaq(sys):
	sys.history.sort(key = lambda i : i.birth_year)
	return sys.history[:npop]

def choose_population_add_elite_1(sys):
	sys.history.sort(key = lambda i : i.birth_year)
	ret = sys.history[:npop]
	sys.history.sort(key = lambda i : i.fitness)
	ret.extend(sys.history[:1])
	return ret

def choose_population_add_elite_21(sys):
	sys.history.sort(key = lambda i : i.birth_year)
	ret = sys.history[:npop]
	sys.history.sort(key = lambda i : i.fitness)
	ret.extend(sys.history[:21])
	return ret

def choose_population_replace_by_elite(sys):
	pass

def init():
	init_rough_gmm()
	max_gradient = 0.0

n = 20
npop = 6 * n
npar = n + 1
nchi = 6 * n
step_count = 27200
loop_count = 1
problem = sphere
raw_problem = sphere
title = '{f}(D{d}), pop{npop},par{npar},chi{nchi},step{s},loop{l}'.format(
	f = problem.__name__, d = n, npop = npop, npar = npar, nchi = nchi, s = step_count, l = loop_count)
best_list = {}
print(title)

for _ in range(loop_count):
	randseed = np.random.randint(0x7fffffff)

	np.random.seed(randseed)
	jgg_sys = JGGSystem(problem, n, npop, npar, nchi)
	jgg_sys.step(step_count)
	jgg_sys.calc_raw_fitness(raw_problem)
	best = jgg_sys.get_best_individual()
	if "jgg" in best_list:
		best_list["jgg"] += best.raw_fitness / loop_count
	else:
		best_list["jgg"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, jgg_sys.history,
				color = 'r', label = 'JGG : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, n, npop, npar, nchi)
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.step(step_count)
	swap_sys.calc_raw_fitness(raw_problem)
	best = swap_sys.get_best_individual()
	if "clamp_rand" in best_list:
		best_list["clamp_rand"] += best.raw_fitness / loop_count
	else:
		best_list["clamp_rand"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, swap_sys.get_active_system().history,
				color = 'b', label = 'clamp_rand : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, n, npop, npar, nchi)
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = choose_population_throw_gaq
	swap_sys.step(step_count)
	swap_sys.calc_raw_fitness(raw_problem)
	best = swap_sys.get_best_individual()
	if "throw_gaq" in best_list:
		best_list["throw_gaq"] += best.raw_fitness / loop_count
	else:
		best_list["throw_gaq"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, swap_sys.get_active_system().history,
				color = 'g', label = 'throw_gaq : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, n, npop, npar, nchi)
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = choose_population_add_elite_1
	swap_sys.step(step_count)
	swap_sys.calc_raw_fitness(raw_problem)
	best = swap_sys.get_best_individual()
	if "add_elite_1" in best_list:
		best_list["add_elite_1"] += best.raw_fitness / loop_count
	else:
		best_list["add_elite_1"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, swap_sys.get_active_system().history,
				color = 'orange', label = 'add_elite_1 : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, n, npop, npar, nchi)
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = choose_population_add_elite_21
	swap_sys.step(step_count)
	swap_sys.calc_raw_fitness(raw_problem)
	best = swap_sys.get_best_individual()
	if "add_elite_21" in best_list:
		best_list["add_elite_21"] += best.raw_fitness / loop_count
	else:
		best_list["add_elite_21"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, swap_sys.get_active_system().history,
				color = 'yellow', label = 'add_elite_21 : {:.10f}'.format(best.raw_fitness))

	if loop_count == 1:
		# plt.axis(xmin = 0, ymin = 0)
		plt.title(title)
		plt.legend()
		plt.show()

for key, ave in best_list.items():
	print(key, ave)
