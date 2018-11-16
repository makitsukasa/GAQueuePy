import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.randn(100) * 0.2

y2 = np.convolve(y, np.ones(5) / 5, mode = 'same')

plt.plot(x, y, 'k-', label = 'original')
plt.plot(x, y2, 'b-', label = 'moving average')
plt.legend()
plt.show()
