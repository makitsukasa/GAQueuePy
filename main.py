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
from plot import plot
import warnings

warnings.simplefilter("error", RuntimeWarning)

def gaq_op_plain(x):
	x.sort(key = lambda i : i.fitness)
	parents = x[:npar]
	for p in parents:
		p.state = State.USED_IN_GAQ
	return crossoverer.rex(parents, nchi)

def gaq_op_plain_origopt(x):
	x.sort(key = lambda i : i.fitness)
	parents = x[:n + 1]
	for p in parents:
		p.state = State.USED_IN_GAQ
	return crossoverer.rex(parents)

def throw_generated(sys):
	sys.history.sort(key = lambda i : i.birth_year)
	return sys.history[:npop]

def throw_random_parents(sys, count):
	np.random.shuffle(sys.history)
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	parents = [i for i in initial if i.state == State.USED_IN_GAQ]
	ret = [i for i in initial if i.state != State.USED_IN_GAQ]
	np.random.shuffle(parents)
	ret.extend(parents[count:])
	return ret

def throw_losed_parents(sys, count):
	np.random.shuffle(sys.history)
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	parents = [i for i in initial if i.state == State.USED_IN_GAQ]
	ret = [i for i in initial if i.state != State.USED_IN_GAQ]
	parents.sort(key = lambda i : i.fitness)
	ret.extend(parents[count:])
	return ret

def throw_random(sys, count):
	np.random.shuffle(sys.history)
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	np.random.shuffle(initial)
	return initial[count:]

def throw_losed(sys, count):
	np.random.shuffle(sys.history)
	sys.history.sort(key = lambda i : i.birth_year)
	initial = sys.history[:npop]
	initial.sort(key = lambda i : i.fitness)
	return initial[count:]

def pick_elites(sys, count):
	sys.history.sort(key = lambda i : i.fitness)
	return sys.history[:count]

def not_replaced(sys):
	return throw_generated(sys)

def replace_random_parents_by_elites(sys, count):
	ret = throw_random_parents(sys, count)
	ret.extend(pick_elites(sys, count))
	return ret

def replace_losed_parents_by_elites(sys, count):
	ret = throw_losed_parents(sys, count)
	ret.extend(pick_elites(sys, count))
	return ret

def replace_random_by_elites(sys, count):
	ret = throw_random(sys, count)
	ret.extend(pick_elites(sys, count))
	return ret

def replace_losed_by_elites(sys, count):
	ret = throw_losed(sys, count)
	ret.extend(pick_elites(sys, count))
	return ret

def init():
	init_rough_gmm()

n = 20

problem_info = {"problem_name" : "sphere", "problem" : sphere, "step" : 27200, "npop" : 6 * n, "nchi" : 6 * n}
# problem_info = {"problem_name" : "ellipsoid", "problem" : ellipsoid, "step" : 33800, "npop" : 6 * n, "nchi" : 6* n}
# problem_info = {"problem_name" : "k-tablet", "problem" : ktablet, "step" : 48000, "npop" : 8 * n, "nchi" : 6 *n}
# problem_info = {"problem_name" : "rosenbrock", "problem" : rosenbrock, "step" : 157000, "npop" : 15 * n, "nchi" : 8 * n}
# problem_info = {"problem_name" : "bohachevsky", "problem" : bohachevsky, "step" : 33800, "npop" : 6 * n, "nchi" : 6 * n}
# problem_info = {"problem_name" : "ackley", "problem" : ackley, "step" : 55400, "npop" : 8 * n, "nchi" : 6 * n}
# problem_info = {"problem_name" : "schaffer", "problem" : schaffer, "step" : 229000, "npop" : 10 * n, "nchi" : 8* n}
# problem_info = {"problem_name" : "rastrigin", "problem" : rastrigin, "step" : 220000, "npop" : 24 * n, "nchi" : 8 * n}

npop = problem_info["npop"]
npar = n + 1
nchi = problem_info["nchi"]
goal = 1e-7
# step_count = 100 * n
step_count = 300000
loop_count = 1
problem = problem_info["problem"]
raw_problem = problem_info["problem"]
title = '{f}(D{d}), pop{npop},par{npar},chi{nchi},step{s},loop{l}'.format(
	f = problem.__name__, d = n, npop = npop, npar = npar, nchi = nchi, s = step_count, l = loop_count)
histories = {}

print(title)

for _ in range(loop_count):
	randseed = np.random.randint(0x7fffffff)

	init()
	np.random.seed(randseed)
	jgg_sys = JGGSystem(problem, raw_problem, n, npop, npar, nchi)
	succeeded = jgg_sys.until_goal(goal, step_count)
	best = jgg_sys.get_best_individual()
	if "JGG" in histories:
		histories["JGG"].append(jgg_sys.history)
	else:
		histories["JGG"] = [jgg_sys.history]

	init()
	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = not_replaced
	succeeded = swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "$R_{入れ替えない}$" in histories:
		histories["$R_{入れ替えない}$"].append(swap_sys.get_active_system().history)
	else:
		histories["$R_{入れ替えない}$"] = [swap_sys.get_active_system().history]

	init()
	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : replace_random_parents_by_elites(sys, npar)
	succeeded = swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "$R_{親全部}$" in histories:
		histories["$R_{親全部}$"].append(swap_sys.get_active_system().history)
	else:
		histories["$R_{親全部}$"] = [swap_sys.get_active_system().history]

	init()
	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : replace_random_parents_by_elites(sys, npar // 3)
	succeeded = swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "$R_{ランダムな親}$" in histories:
		histories["$R_{ランダムな親}$"].append(swap_sys.get_active_system().history)
	else:
		histories["$R_{ランダムな親}$"] = [swap_sys.get_active_system().history]

	init()
	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : replace_losed_parents_by_elites(sys, npar // 3)
	succeeded = swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "$R_{劣った親}$" in histories:
		histories["$R_{劣った親}$"].append(swap_sys.get_active_system().history)
	else:
		histories["$R_{劣った親}$"] = [swap_sys.get_active_system().history]

	init()
	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : replace_random_by_elites(sys, npar)
	succeeded = swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "$R_{ランダム}$" in histories:
		histories["$R_{ランダム}$"].append(swap_sys.get_active_system().history)
	else:
		histories["$R_{ランダム}$"] = [swap_sys.get_active_system().history]

	init()
	np.random.seed(randseed)
	swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
	swap_sys.gaq_sys.op = gaq_op_plain_origopt
	swap_sys.switch_to_gaq = lambda sys : False
	swap_sys.choose_population_to_jgg = lambda sys : replace_losed_by_elites(sys, npar)
	succeeded = swap_sys.until_goal(goal, step_count)
	best = swap_sys.get_best_individual()
	if "$R_{劣った}$" in histories:
		histories["$R_{劣った}$"].append(swap_sys.get_active_system().history)
	else:
		histories["$R_{劣った}$"] = [swap_sys.get_active_system().history]

if loop_count == 1:
	color_dict = {
		"JGG" : "r",
		"$R_{入れ替えない}$" : "gray",
		"$R_{親全部}$" : "yellow",
		"$R_{ランダムな親}$" : "orange",
		"$R_{劣った親}$" : "yellowgreen",
		"$R_{ランダム}$" : "b",
		"$R_{劣った}$" : "green",
	}
	for method_name in histories.keys():
		his = histories[method_name][0]
		plot(len(his), his, color = color_dict[method_name], label = method_name)
	plt.axis(xmin = 0, ymin = 0)
	# plt.title(title)
	plt.xlabel("目的関数の評価回数")
	plt.ylabel("目的関数の評価値")
	plt.legend()
	plt.show()
