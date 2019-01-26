import numpy as np
import matplotlib.pyplot as plt
from jggsystem import JGGSystem
from gaqsystem import GAQSystem
from swapsystem import SwapSystem
from swapsystem2 import SwapSystem2
from individual import Individual, State
import crossoverer
from problem.frontier.sphere import sphere
from problem.frontier.ellipsoid import ellipsoid
from problem.frontier.ackley import ackley
from problem.frontier.rastrigin import rastrigin
from problem.frontier.ktablet import ktablet
from problem.frontier.rosenbrock import rosenbrock
from problem.frontier.bohachevsky import bohachevsky
from problem.frontier.schaffer import schaffer
from problem.gmm.gmm import gmm, rough_gmm_ave, rough_gmm_weighted_ave, init_rough_gmm
from plot import plot, plot_error
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

def choose_population_replace_parents_by_elites(sys, elites_count):
	np.random.shuffle(sys.history)
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	parents = [i for i in initial if i.state == State.USED_IN_GAQ]
	unmarried = [i for i in initial if i.state != State.USED_IN_GAQ]
	ret = unmarried
	ret.extend(parents[elites_count:])
	sys.history.sort(key = lambda i : i.fitness)
	ret.extend(sys.history[:elites_count])
	return ret

def choose_population_replace_bottom_by_elites(sys, elites_count):
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	np.random.shuffle(sys.history)
	initial.sort(key = lambda i : -i.fitness)
	ret = initial[elites_count:]
	sys.history.sort(key = lambda i : i.fitness)
	ret.extend(sys.history[:elites_count])
	return ret

def choose_population_replace_random_by_elites(sys, elites_count):
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	np.random.shuffle(sys.history)
	ret = initial[elites_count:]
	sys.history.sort(key = lambda i : i.fitness)
	ret.extend(sys.history[:elites_count])
	return ret

def init():
	init_rough_gmm()
	max_gradient = 0.0

n = 20
npop = 8 * n
npar = n + 1
nchi = 6 * n
goal = 1e-7
step_count = 250000
loop_count = 3
problem = sphere
raw_problem = sphere
title = '{f}(D{d}), pop{npop},par{npar},chi{nchi},step{s},loop{l}'.format(
	f = problem.__name__, d = n, npop = npop, npar = npar, nchi = nchi, s = step_count, l = loop_count)
best_list = {}
step_list = {}
histories = {"jgg" : [], "throw_gaq" : [], "replace" : []}
print(title)

for _ in range(loop_count):
	randseed = np.random.randint(0x7fffffff)

	np.random.seed(randseed)
	jgg_sys = JGGSystem(problem, raw_problem, n, npop, npar, nchi)
	reached = jgg_sys.until_goal(goal, step_count)
	best = jgg_sys.get_best_individual()
	if "jgg" in best_list:
		best_list["jgg"] += best.raw_fitness / loop_count
		step_list["jgg"] += len(jgg_sys.history) / loop_count
	else:
		best_list["jgg"] = best.raw_fitness / loop_count
		step_list["jgg"] = len(jgg_sys.history) / loop_count
	histories["jgg"].append(jgg_sys.history)

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = choose_population_throw_gaq
	swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "throw_gaq" in best_list:
		best_list["throw_gaq"] += best.raw_fitness / loop_count
		step_list["throw_gaq"] += len(swap_sys.get_active_system().history) / loop_count
	else:
		best_list["throw_gaq"] = best.raw_fitness / loop_count
		step_list["throw_gaq"] = len(swap_sys.get_active_system().history) / loop_count
	histories["throw_gaq"].append(jgg_sys.history)

	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : choose_population_replace_parents_by_elites(sys, npar // 3)
	swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "replace" in best_list:
		best_list["replace"] += best.raw_fitness / loop_count
		step_list["replace"] += len(swap_sys.get_active_system().history) / loop_count
	else:
		best_list["replace"] = best.raw_fitness / loop_count
		step_list["replace"] = len(swap_sys.get_active_system().history) / loop_count
	histories["replace"].append(jgg_sys.history)

for method in histories:
	plot_error(histories[method])

# plt.axis(xmin = 0, ymin = 0)
plt.title(title)
plt.legend()
plt.show()

for key, ave in best_list.items():
	print(key, step_list[key], ave)
