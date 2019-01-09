import numpy as np
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
from problem.gmm.gmm import gmm, rough_gmm_ave, rough_gmm_weighted_ave, rough_gmm_compared, init_rough_gmm
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

best_lists = {}
problem = lambda x : rough_gmm_ave(x, 0.5)
problem_name = "gmm ave 0.5"
raw_problem = gmm
npop = 20
npar = 5
nchi = 20
step_count = 400
loop_count = 30

for n in [3, 5, 7, 10]:

	best_list = {}

	for _ in range(loop_count):
		randseed = np.random.randint(0x7fffffff)

		init()
		np.random.seed(randseed)
		jgg_sys = JGGSystem(problem, n, npop, npar, nchi)
		jgg_sys.step(step_count)
		jgg_sys.calc_raw_fitness(raw_problem)
		best = jgg_sys.get_best_individual()
		if "jgg" in best_list:
			best_list["jgg"] += best.raw_fitness / loop_count
		else:
			best_list["jgg"] = best.raw_fitness / loop_count

		init()
		np.random.seed(randseed)
		gaq_sys = GAQSystem(problem, 0, [Individual(n) for i in range(npop)], gaq_op_plain_origopt)
		gaq_sys.step(step_count)
		gaq_sys.calc_raw_fitness(raw_problem)
		best = gaq_sys.get_best_individual()
		if "gaq" in best_list:
			best_list["gaq"] += best.raw_fitness / loop_count
		else:
			best_list["gaq"] = best.raw_fitness / loop_count

		init()
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

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem(problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : choose_population_replace_by_elites(sys, npar)
		swap_sys.step(step_count)
		swap_sys.calc_raw_fitness(raw_problem)
		best = swap_sys.get_best_individual()
		if "replace" in best_list:
			best_list["replace"] += best.raw_fitness / loop_count
		else:
			best_list["replace"] = best.raw_fitness / loop_count

		init()
		np.random.seed(randseed)
		swap_sys = SwapSystem2(problem, n, npop, npar, nchi)
		swap_sys.gaq_sys.op = gaq_op_plain_origopt
		swap_sys.switch_to_gaq = lambda sys : False
		swap_sys.choose_population_to_jgg = lambda sys : choose_population_replace_by_elites(sys, npar)
		swap_sys.step(step_count)
		swap_sys.calc_raw_fitness(raw_problem)
		best = swap_sys.get_best_individual()
		if "replace_2" in best_list:
			best_list["replace_2"] += best.raw_fitness / loop_count
		else:
			best_list["replace_2"] = best.raw_fitness / loop_count

	best_lists[n] = best_list

print(problem_name, "|3D|5D|7D|10D|")
print("|:--|:--|:--|:--|:--|")
print("(論文)|-5.71|-3.42|-2.98|-2.04|")

for solution in best_lists[3].keys():
	print(solution, "|", end = "")
	for dim in [3, 5, 7, 10]:
		print(round(best_lists[dim][solution], 2), "|", end = "")
	print()
