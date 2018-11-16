import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

font = {"family":"Noto Sans JP Regular"}
mpl.rc('font', **font)

def plot(step_count, history, fmt = 'b-', label = ''):
	x = range(step_count)
	v = step_count // 100
	np.random.shuffle(history)
	history.sort(key = lambda i: i.birth_year if i.birth_year is not None else -1)
	f_raw = [i.fitness for i in history]
	f = np.convolve(f_raw, np.ones(v) / v, mode = 'vaild')
	plt.plot(x[:len(f)], f, fmt, label = label)
