import matplotlib.pyplot as plt
import numpy as np
filename = '24Hz_l.csv'
filename_r = '24Hz_r.csv'
data = np.genfromtxt(filename, delimiter=',')
data_r = np.genfromtxt(filename_r, delimiter=',')
fig, ax = plt.subplots(1)
plt.plot(data[:, 0], data[:, 1])

plt.xlabel('frequency (Hz)')
plt.ylabel('Power (dB)')
y_max_index = np.argmax(data[:, 1])
label_max = "402 Hz"
plt.annotate(label_max, (data[y_max_index, 0], data[y_max_index, 1]),
                                       textcoords="offset points",
                                       xytext=(10, -5),
                                       ha='left')
# plt.savefig('24_Hz_entrainment_Left.pdf')

plt.plot(data_r[:, 0], data_r[:, 1])
plt.plot(data[y_max_index, 0], data[y_max_index, 1], 'r.')
# plt.set(xlim=[0, 490])
# plt.set(xlabel='frequency (Hz)', ylabel='Power (dB)')
plt.xlabel('frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xlim([0, 490])
# plt.set(xlabel='frequency (Hz)', ylabel='Power (dB)')

y_max_index = np.argmax(data_r[:, 1])
label_max = "426 Hz"
plt.annotate(label_max, (data_r[y_max_index, 0], data_r[y_max_index, 1]),
                                       textcoords="offset points",
                                       xytext=(10, -5),
                                       ha='left')
plt.plot(data_r[y_max_index, 0], data_r[y_max_index, 1], 'r.')
plt.legend(['Left Channel', 'Right Channel'])
plt.savefig('24_Hz_entrainment.pdf')

filename = '18Hz_l.csv'
filename_r = '18Hz_r.csv'
data = np.genfromtxt(filename, delimiter=',')
data_r = np.genfromtxt(filename_r, delimiter=',')

fig2, ax = plt.subplots(1)
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xlim([0, 490])
# plt.set(xlim=[0, 490])
# plt.set(xlabel='frequency (Hz)', ylabel='Power (dB)')
# plt.set(xlabel='frequency (Hz)', ylabel='Power (dB)')
y_max_index = np.argmax(data[:, 1])
label_max = "402 Hz"
plt.annotate(label_max, (data[y_max_index, 0], data[y_max_index, 1]),
                                       textcoords="offset points",
                                       xytext=(10, -5),
                                       ha='left')
plt.plot(data_r[:, 0], data_r[:, 1])
plt.plot(data[y_max_index, 0], data[y_max_index, 1], 'r.')

# plt.savefig('18_Hz_entrainment_left.pdf')

plt.legend(['Left Channel', 'Right Channel'])
# plt.set(xlim=[0, 490])

y_max_index = np.argmax(data_r[:, 1])
label_max = "422 Hz"
plt.annotate(label_max, (data_r[y_max_index, 0], data_r[y_max_index, 1]),
                                       textcoords="offset points",
                                       xytext=(10, -5),
                                       ha='left')
plt.plot(data_r[y_max_index, 0], data_r[y_max_index, 1], 'r.')
plt.savefig('18_Hz_entrainment.pdf')

filename = 'pink_ours.csv'
filename_r = 'pink_ideal.csv'
data = np.genfromtxt(filename, delimiter=',')
data_r = np.genfromtxt(filename_r, delimiter=',')

fig3, ax = plt.subplots(1)
# plt.plot(data[:, 0], data[:, 1])
# plt.set(xlim=[0, 20000])


plt.plot(data_r[:, 0], data_r[:, 1])
# plt.set(xlim=[0, 20000])
# plt.set(xlabel='frequency (Hz)', ylabel='Power (dB)')
# plt.set(xlabel='frequency (Hz)', ylabel='Power (dB)')
plt.xlabel('frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xlim([0, 20000])
plt.savefig('pink.pdf')

# print(data[:,0])
