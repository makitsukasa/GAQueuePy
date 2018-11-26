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

def rough_gmm_ave(x):
	global count
	global mean
	val = gmm(x)
	mean = (mean * count + val) / (count + 1)
	count += 1
	if abs(mean - val) < 0.01:
		return 0.0
	elif mean < val:
		return 1.0
	else:
		return -1.0

# approx : sum(k:0->n) r**k is 2
def rough_gmm_disc(x):
	global disc_sum
	disc_rate = 0.2
	val = gmm(x)
	ret = 0.0
	new_disc_sum = (val + disc_sum * 2 * disc_rate) / 2
	if abs(new_disc_sum - disc_sum) > 0.01:
		if new_disc_sum > disc_sum:
			ret = 1.0
		else:
			ret = -1.0
	disc_sum = new_disc_sum
	return ret
