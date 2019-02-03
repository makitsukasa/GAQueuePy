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

def init():
	init_rough_gmm()

best_lists = {}
step_lists = {}
league = {}
for team in ["$R_{入れ替えない}$", "$R_{ランダムな親}$"]:
	league[team] = {}
	for enem in ["$R_{入れ替えない}$", "$R_{ランダムな親}$"]:
		league[team][enem] = 0

n = 20
npar = n + 1
loop_count = 30
goal = 1e-7
problem_list = [
	{"problem_name" : "sphere", "problem" : sphere, "step" : 27200, "npop" : 6 * n, "nchi" : 6 * n},
	# # {"problem_name" : "ellipsoid", "problem" : ellipsoid, "step" : 33800, "npop" : 6 * n, "nchi" : 6 * n},
	# {"problem_name" : "k-tablet", "problem" : ktablet, "step" : 48000, "npop" : 8 * n, "nchi" : 6 * n},
	# # {"problem_name" : "rosenbrock", "problem" : rosenbrock, "step" : 157000, "npop" : 15 * n, "nchi" : 8 * n},
	# {"problem_name" : "bohachevsky", "problem" : bohachevsky, "step" : 33800, "npop" : 6 * n, "nchi" : 6 * n},
	# {"problem_name" : "ackley", "problem" : ackley, "step" : 55400, "npop" : 8 * n, "nchi" : 6 * n},
	# {"problem_name" : "schaffer", "problem" : schaffer, "step" : 229000, "npop" : 10 * n, "nchi" : 8 * n},
	# {"problem_name" : "rastrigin", "problem" : rastrigin, "step" : 220000, "npop" : 24 * n, "nchi" : 8 * n},
]

for problem_info in problem_list:

	best_list = {}
	step_list = {}
	league_score = {}
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
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = choose_population_throw_gaq
		swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if "$R_{入れ替えない}$" in best_list:
			best_list["$R_{入れ替えない}$"] += best.raw_fitness / loop_count
			step_list["$R_{入れ替えない}$"] += float(len(swap_sys.get_active_system().history)) / loop_count
		else:
			best_list["$R_{入れ替えない}$"] = best.raw_fitness / loop_count
			step_list["$R_{入れ替えない}$"] = float(len(swap_sys.get_active_system().history)) / loop_count
		league_score["$R_{入れ替えない}$"] = len(swap_sys.get_active_system().history)

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, raw_problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : choose_population_replace_parents_by_elites(sys, npar // 3)
		swap_sys.until_goal(goal, step_count)
		best = swap_sys.get_best_individual()
		if "$R_{ランダムな親}$" in best_list:
			best_list["$R_{ランダムな親}$"] += best.raw_fitness / loop_count
			step_list["$R_{ランダムな親}$"] += float(len(swap_sys.get_active_system().history)) / loop_count
		else:
			best_list["$R_{ランダムな親}$"] = best.raw_fitness / loop_count
			step_list["$R_{ランダムな親}$"] = float(len(swap_sys.get_active_system().history)) / loop_count
		league_score["$R_{ランダムな親}$"] = len(swap_sys.get_active_system().history)

		for team in league_score:
			for enem in league_score:
				if team == enem:
					continue
				elif league_score[team] < league_score[enem]:
					league[team][enem] += 1

	best_lists[problem_name] = best_list
	step_lists[problem_name] = step_list

method_names = list(list(best_lists.values())[0].keys())

print("|", end = "")
for team in league:
	print("|", team, end = "")
print("|")
print("|--:|--:|--:|--:|")
for team in league:
	print(team, end = "")
	for enem in league:
		if team == enem:
			print("|-", end = "")
		else:
			print("|", league[team][enem], end = "")
	print("|")
