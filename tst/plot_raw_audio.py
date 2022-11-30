import matplotlib.pyplot as plt
import numpy as np

filename = '24Hz_l_raw.csv'
filename_r = '24Hz_r_raw.csv'
data = np.genfromtxt(filename, delimiter=',')
data_r = np.genfromtxt(filename_r, delimiter=',')
fig, ax = plt.subplots(1)

data_length = range(0, len(data))

time = list(map(lambda x: x/512, data_length))

plt.plot(time, data)
plt.plot(time, data_r)

plt.xlabel('time (s)')
plt.ylabel('recorded audio (V)')
plt.legend(['Left Channel', 'Right Channel'])
plt.tight_layout()
plt.savefig('24_Hz_entrainment_raw.pdf')

filename = '18Hz_l_raw.csv'
filename_r = '18Hz_r_raw.csv'
# data = np.genfromtxt(filename, delimiter=',')
data_r = np.genfromtxt(filename_r, delimiter=',')
fig, ax = plt.subplots(1)

data_length = range(0, len(data))

time = list(map(lambda x: x/512, data_length))

plt.plot(time, data)
plt.plot(time, data_r)

plt.xlabel('time (s)')
plt.ylabel('recorded audio (V)')
plt.legend(['Left Channel', 'Right Channel'])
plt.tight_layout()
plt.savefig('18_Hz_entrainment_raw.pdf')
