import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import matplotlib as mpl

available_fonts = fm.findSystemFonts()
print(available_fonts)

font = {"family": "Noto Sans JP Regular"}
mpl.rc('font', **font)
plt.rcParams["mathtext.default"] = "regular"

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
