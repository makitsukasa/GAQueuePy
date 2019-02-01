import numpy as np
from jggsystem import JGGSystem
from gaqsystem import GAQSystem
from swapsystem import SwapSystem
from swapsystem2 import SwapSystem2
from individual import Individual, State
import crossoverer
from problem.frontier.sphere import sphere
from problem.frontier.ellipsoid import ellipsoid
from problem.frontier.ktablet import ktablet
from problem.frontier.rosenbrock import rosenbrock
from problem.frontier.bohachevsky import bohachevsky
from problem.frontier.ackley import ackley
from problem.frontier.schaffer import schaffer
from problem.frontier.rastrigin import rastrigin
from problem.gmm.gmm import gmm, rough_gmm_ave, rough_gmm_weighted_ave, init_rough_gmm
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

histories_list = {}

n = 20
npar = n + 1
loop_count = 1
goal = 1e-7
problem_list = [
	# {"problem_name" : "sphere", "problem" : sphere, "step" : 27200, "npop" : 6 * n, "nchi" : 6 * n},
	# {"problem_name" : "ellipsoid", "problem" : ellipsoid, "step" : 33800, "npop" : 6 * n, "nchi" : 6 * n},
	# {"problem_name" : "k-tablet", "problem" : ktablet, "step" : 48000, "npop" : 8 * n, "nchi" : 6 * n},
	# {"problem_name" : "rosenbrock", "problem" : rosenbrock, "step" : 157000, "npop" : 15 * n, "nchi" : 8 * n},
	# {"problem_name" : "bohachevsky", "problem" : bohachevsky, "step" : 33800, "npop" : 6 * n, "nchi" : 6 * n},
	# {"problem_name" : "ackley", "problem" : ackley, "step" : 55400, "npop" : 8 * n, "nchi" : 6 * n},
	{"problem_name" : "schaffer", "problem" : schaffer, "step" : 229000, "npop" : 10 * n, "nchi" : 8 * n},
	# {"problem_name" : "rastrigin", "problem" : rastrigin, "step" : 220000, "npop" : 24 * n, "nchi" : 8 * n},
]

for problem_info in problem_list:

	histories = {}
	problem = problem_info["problem"]
	raw_problem = problem_info["problem"]
	problem_name = problem_info["problem_name"]
	npop = problem_info["npop"]
	nchi = problem_info["nchi"]
	# step_count = problem_info["step"]
	# step_count = problem_info["step"] // 10
	# step_count = 100 * n
	step_count = 300000
	print(problem_name, "step:", step_count)

	for _ in range(loop_count):
		randseed = np.random.randint(0x7fffffff)

		init()
		np.random.seed(randseed)
		jgg_sys = JGGSystem(problem, raw_problem, n, npop, npar, nchi)
		succeeded = jgg_sys.until_goal(goal, step_count)
		best = jgg_sys.get_best_individual()
		if succeeded:
			if "JGG" in histories:
				histories["JGG"].append(jgg_sys.history)
			else:
				histories["JGG"] = [jgg_sys.history]
		else:
			print("JGG failed")

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = not_replaced
		succeeded = swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if succeeded:
			if "$R_{入れ替えない}$" in histories:
				histories["$R_{入れ替えない}$"].append(swap_sys.get_active_system().history)
			else:
				histories["$R_{入れ替えない}$"] = [swap_sys.get_active_system().history]
		else:
			print("$R_{入れ替えない}$ failed")

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : replace_random_parents_by_elites(sys, npar)
		succeeded = swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if succeeded:
			if "$R_{親全部}$" in histories:
				histories["$R_{親全部}$"].append(swap_sys.get_active_system().history)
			else:
				histories["$R_{親全部}$"] = [swap_sys.get_active_system().history]
		else:
			print("$R_{親全部}$ failed")

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : replace_random_parents_by_elites(sys, npar // 3)
		succeeded = swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if succeeded:
			if "$R_{ランダムな親}$" in histories:
				histories["$R_{ランダムな親}$"].append(swap_sys.get_active_system().history)
			else:
				histories["$R_{ランダムな親}$"] = [swap_sys.get_active_system().history]
		else:
			print("$R_{ランダムな親}$ failed")

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : replace_losed_parents_by_elites(sys, npar // 3)
		succeeded = swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if succeeded:
			if "$R_{劣った親}$" in histories:
				histories["$R_{劣った親}$"].append(swap_sys.get_active_system().history)
			else:
				histories["$R_{劣った親}$"] = [swap_sys.get_active_system().history]
		else:
			print("$R_{劣った親}$ failed")

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : replace_random_by_elites(sys, npar)
		succeeded = swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if succeeded:
			if "$R_{ランダム}$" in histories:
				histories["$R_{ランダム}$"].append(swap_sys.get_active_system().history)
			else:
				histories["$R_{ランダム}$"] = [swap_sys.get_active_system().history]
		else:
			print("$R_{ランダム}$ failed")

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : replace_losed_by_elites(sys, npar)
		succeeded = swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if succeeded:
			if "$R_{劣った}$" in histories:
				histories["$R_{劣った}$"].append(swap_sys.get_active_system().history)
			else:
				histories["$R_{劣った}$"] = [swap_sys.get_active_system().history]
		else:
			print("$R_{劣った}$ failed")

	histories_list[problem_name] = histories

print("loop", loop_count)

problem_names = list(histories_list.keys())
method_names = list(list(histories_list.values())[0].keys())
print("\\begin{table}[p]\\begin{center}")
print("\\caption{}")
print("\\label{}")
print("\\begin{tabular}{|l|", end = "")
print("r|" * len(method_names), end = "")
print("}\\hline")
print("&", end = "")
line = ""
for method_name in method_names:
	line += method_name + "&"
print(line[:-1], "\\\\ \\hline")

for problem_name in problem_names:
	print(problem_name, "&", end = "")
	line = ""
	for method_name in method_names:
		# print(int(round(bests[method_name] * 100, 0)), "/",
		# 	step_lists[problem_name][method_name], "|", end = "")
		# print(step_lists[problem_name][method_name], "|", end = "")
		ave_step = np.average([len(history) for history in histories_list[problem_name][method_name]])
		line += str(int(round(ave_step))) + "&"
		# print(int(round(bests[method_name] * 100, 0)), "|", end = "")
	print(line[:-1], "\\\\ \\hline")

print("\\end{tabular}")
print("\\end{center}\\end{table}")
