import numpy as np
from math import sin
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy import signal
import pandas as pd

N = 512 * 3
ix = np.arange(N)
ix = ix / 512
# print(ix)
signal1 = 3*np.sin(2 * np.pi * ix * 20)
#
signal2 = np.sin(2 * np.pi * ix * 35)

signal3 = np.sin(2 * np.pi * ix * 1)

signal4 = np.sin(2 * np.pi * ix * 9)

signal5 = np.sin(2 * np.pi * ix * 5)

time = ix
compound_signal = signal1 + signal2 + signal3 + signal4 + signal5
#

fs = 512
nyq = 0.5 * fs
normal_cutoff = 1 / nyq
b, a = signal.butter(4, [0.001, 100], fs=fs, analog=False, btype='band')
filtered = signal.filtfilt(b, a, compound_signal)

output_array = np.zeros([1, 64])
# for i in range(1, N):
#     row = []
#     for j in range(64):
#         row.append(compound_signal[i])
#     output_array = np.append(output_array, [row], axis=0)
# np.savetxt('sinTest.csv', output_array, delimiter=",")

plt.plot(time, filtered)
plt.show()
#
fft_data = fft(compound_signal)
N = len(fft_data)
n = np.arange(N)
T = N / 512
freq = n / T
fig, ax = plt.subplots()
ax.stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
ax.set(xlim=[0, 105])
plt.show()

eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

absolute_fft_values = np.absolute(fft_data)

sample_frequencies = np.fft.rfftfreq(len(compound_signal), 1.0 / fs)

eeg_band_fft = dict()
for band in eeg_bands:
    band_frequency_values = np.where((sample_frequencies >= eeg_bands[band][0]) &
                                     (sample_frequencies <= eeg_bands[band][1]))[0]
    eeg_band_fft[band] = np.max(absolute_fft_values[band_frequency_values])  # np.mean gives a skewed perception here


df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
my_colors = [(0.50, x/4.0, x/5.0) for x in range(len(df))]
ax = df.plot.bar(x='band', y='val', legend=False, color=my_colors)
ax.set_xlabel("EEG band")
ax.set_ylabel("Maximum band Amplitude")
plt.show()
