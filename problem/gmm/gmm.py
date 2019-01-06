import numpy as np

MU = [
	[-1.0,  1.5, -2.0,  2.5],
	[ 0.0, -2.0,  3.0,  1.0],
	[-2.5, -2.0,  1.5,  3.5],
	[-2.0,  1.0, -1.0,  3.0],
]
SQ_SIGMA = [ 2.25, 4.0, 1.0, 4.0 ]
A = [ 3.1, 3.4, 4.1, 3.0 ]

def gmm(x):
	shifted = x * 10.24 - 5.12
	ans = 0.0
	for i in range(4):
		exponent = 0.0
		for j in range(len(shifted)):
			exponent -= ((shifted[j] - MU[i][j % 4]) ** 2) / (2 * SQ_SIGMA[i])
		ans += A[i] * np.exp(exponent)
	return -ans

count = 0
mean = 0.0
disc_sum = 0.0
history = []
landmark = None

def init_rough_gmm():
	global count
	global mean
	global disc_sum
	global history
	global landmark
	count = 0
	mean = 0.0
	disc_sum = 0.0
	history = []
	landmark = {"x" : [], "fitness" : 0, "raw_fitness" : 0.0}

def rough_gmm_ave(x, magnification = 1.0):
	global count
	global mean
	val = gmm(x)
	ret = -1.0
	if abs(mean * magnification - val) < 0.01:
		ret = 0.0
	elif mean * magnification < val:
		ret = 1.0
	mean = (mean * count + val) / (count + 1)
	count += 1
	return ret

def rough_gmm_weighted_ave(x, rate = 0.5):
	global disc_sum
	global count
	val = gmm(x)
	ret = -1.0
	if abs(val - disc_sum) < 0.01:
		ret = 0.0
	elif val > disc_sum:
		ret = 1.0
	disc_sum = ((1 - rate) * val + rate * disc_sum * (1 - rate ** count)) / (1 - rate ** (count + 1))
	count += 1
	return ret

def rough_gmm_compared(x, landmark_pos = 5):
	global history
	global landmark

	if not history:
		landmark = {"x" : x, "fitness" : 0, "raw_fitness" : gmm(x)}
		history.append(landmark)
		return 0

	raw_fitness = gmm(x)

	is_better_than_landmark = False
	if abs(landmark["raw_fitness"] - raw_fitness) < 0.01:
		if np.random.rand() < 0.5:
			is_better_than_landmark = True
	elif landmark["raw_fitness"] > raw_fitness:
		is_better_than_landmark = True

	if is_better_than_landmark:
		fitness = landmark["fitness"] - 1
	else:
		fitness = landmark["fitness"] + 1

	history.append({"x" : x, "fitness" : fitness, "raw_fitness" : raw_fitness})

	np.random.shuffle(history)
	history.sort(key = lambda h: h["fitness"])

	if len(history) <= landmark_pos:
		landmark = history[0]
	else:
		landmark = history[landmark_pos]

	return fitness
