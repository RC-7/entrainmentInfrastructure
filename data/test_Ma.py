import numpy as np
import matplotlib.pyplot as plt

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


x = np.ones(100)
out = moving_average(x, 20)

plt.plot(out)
