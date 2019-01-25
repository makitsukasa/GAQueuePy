import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from statistics import mean, stdev

# font = {"family":"Noto Sans JP Regular"}
# mpl.rc('font', **font)

def plot(step_count, history, color = 'b', linestyle = '-', label = ''):
	v = step_count // 100
	np.random.shuffle(history)
	history.sort(key = lambda i: i.birth_year if i.birth_year is not None else -1)
	f_raw = [i.raw_fitness for i in history]
	f = np.convolve(f_raw, np.ones(v) / v, mode = 'vaild')
	x = range(len(f))
	plt.yscale("log")
	plt.plot(x[:len(f)], f, linewidth = 0.5, color = color, linestyle = linestyle, label = label)
	# plt.plot(x[:len(f_raw)], f_raw, linewidth = 0.5, color = color, label = label)

def plot_error(histories, color = 'b', linestyle = '-', label = ''):
	v = 0
	for history in histories:
		np.random.shuffle(history)
		history.sort(key = lambda i: i.birth_year if i.birth_year is not None else -1)
		v = max(len(history), v)
	histories_T = np.array(histories).transpose()
	raw_data = []
	for period in histories_T:
		fitnesses = []
		for i in period:
			fitnesses.append(i.raw_fitness)
		raw_data = mean(fitnesses)
		raw_error = stdev(fitnesses)
	error = np.convolve(raw_error, np.ones(v) / v, mode = 'vaild')
	data = np.convolve(raw_data, np.ones(v) / v, mode = 'vaild')
	x = range(len(data))
	plt.yscale("log")
	# plt.plot(x[:len(data)], data, linewidth = 0.5, color = color, linestyle = linestyle, label = label)
	plt.errorbar(x[:len(data)], data, yerr = error, linewidth = 0.5, color = color, linestyle = linestyle, label = label)
	# plt.plot(x[:len(raw_data)], raw_data, linewidth = 0.5, color = color, label = label)
