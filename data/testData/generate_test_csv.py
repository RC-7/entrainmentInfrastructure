import numpy as np
from math import sin
import matplotlib.pyplot as plt
from numpy.fft import fft

# output_array = []
# increment = 1/512
# for i in range(512):
#     row = []
#     for j in range(64):
#         row.append(sin(increment*i))
#     output_array = np.append(output_array, row, axis=1)

N = 15360
ix = np.arange(N)
ix = ix/512
# print(ix)
signal = np.sin(2*np.pi*ix*20)
#
signal2 = np.sin(2*np.pi*ix*200)
time = ix/512
compound_signal = signal + signal2
#
output_array = np.zeros([1, 64])
for i in range(1, N):
    row = []
    for j in range(64):
        row.append(compound_signal[i])
    output_array = np.append(output_array, [row], axis=0)
np.savetxt('sinTest.csv', output_array, delimiter=",")

# plt.plot(time, compound_signal)
# plt.show()
#
fft_data = fft(compound_signal)
N = len(fft_data)
n = np.arange(N)
T = N / 512
freq = n / T
fig, ax = plt.subplots()
ax.stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
ax.set(xlim=[15, 210])
plt.show()
