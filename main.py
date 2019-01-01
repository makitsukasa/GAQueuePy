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
	parents = x[:n + 1]
	for p in parents:
		p.state = State.USED_IN_GAQ
	return crossoverer.rex(parents)

def choose_population_throw_gaq(sys):
	sys.history.sort(key = lambda i : i.birth_year)
	return sys.history[:npop]

def choose_population_replace_by_elites(sys, elites_count):
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	np.random.shuffle(sys.history)
	ret = []
	for i in initial:
		if elites_count > 0 and i.state == State.USED_IN_GAQ:
			elites_count -= 1
		else:
			ret.append(i)
	sys.history.sort(key = lambda i : i.fitness)
	ret.extend(sys.history[:elites_count])
	return ret

def init():
	init_rough_gmm()
	max_gradient = 0.0

n = 20
npop = 12 * n
npar = n + 1
nchi = 8 * n
step_count = 220000
loop_count = 1
problem = rastrigin
raw_problem = rastrigin
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
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.step(step_count)
	swap_sys.calc_raw_fitness(raw_problem)
	best = swap_sys.get_best_individual()
	if "clamp_rand" in best_list:
		best_list["clamp_rand"] += best.raw_fitness / loop_count
	else:
		best_list["clamp_rand"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, swap_sys.get_active_system().history,
				color = 'black', label = 'clamp_rand : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
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
				color = 'gray', label = 'throw_gaq : {:.10f}'.format(best.raw_fitness))

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : choose_population_replace_by_elites(sys, npar)
	swap_sys.step(step_count)
	swap_sys.calc_raw_fitness(raw_problem)
	best = swap_sys.get_best_individual()
	if "replace_by_elite_npar" in best_list:
		best_list["replace_by_elite_npar"] += best.raw_fitness / loop_count
	else:
		best_list["replace_by_elite_npar"] = best.raw_fitness / loop_count
	if loop_count == 1:
		plot(step_count, swap_sys.get_active_system().history,
				color = 'orange', label = 'replace_by_elite_npar : {:.10f}'.format(best.raw_fitness))

	if loop_count == 1:
		# plt.axis(xmin = 0, ymin = 0)
		plt.title(title)
		plt.legend()
		plt.show()

for key, ave in best_list.items():
	print(key, ave)
