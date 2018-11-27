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

def init_rough_gmm():
	global count
	global mean
	global disc_sum
	count = 0
	mean = 0.0
	disc_sum = 0.0

def rough_gmm_ave(x):
	global count
	global mean
	val = gmm(x)
	ret = -1.0
	if abs(mean - val) < 0.01:
		ret = 0.0
	elif mean < val:
		ret = 1.0
	mean = (mean * count + val) / (count + 1)
	count += 1
	return ret

def rough_gmm_weighted_ave(x):
	global disc_sum
	global count
	r = 0.2
	val = gmm(x)
	ret = -1.0
	if abs(val - disc_sum) < 0.01:
		ret = 0.0
	elif val > disc_sum:
		ret = 1.0
	disc_sum = ((1 - r) * val + r * disc_sum * (1 - r ** count)) / (1 - r ** (count + 1))
	count += 1
	return ret
