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
	x.sort(key=lambda i: i.fitness)
	return crossoverer.rex(x[:n + 1])

def init():
	init_rough_gmm()
	max_gradient = 0.0

n = 20
npop = 6 * n
npar = n + 1
nchi = 6 * n
step_count = 200
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
		plot(step_count, jgg_sys.history, color = 'r', label = 'JGG : {:.10f}'.format(best.raw_fitness))

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
		plot(step_count, swap_sys.get_active_system().history, color = 'y', label = 'clamp_rand : {:.10f}'.format(best.raw_fitness))

	if loop_count == 1:
		# plt.axis(xmin = 0, ymin = 0)
		plt.title(title)
		plt.legend()
		plt.show()

for key, ave in best_list.items():
	print(key, ave)
