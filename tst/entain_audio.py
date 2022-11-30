from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np

# Number of sample points
N = 2000
# sample spacing
sr = 10000
T = 1.0 / sr
t = np.arange(0,1,N*T)
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(100*2*np.pi*x)
y2 = np.sin(150*2*np.pi*x)
yf = fft(y)
yf2 = fft(y2)
N = len(yf)
n = np.arange(N)
T = N/sr
freq = n/T
xf = fftfreq(N, T)[:N//2]
# plt.subplot(1,2,1)
plt.plot(x, y, 'b', label='right ear')
plt.plot(x, y2, 'g', label='left ear')
plt.xlim(0, 0.02)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig('entrain_time_lit.pdf')
plt.close()
# plt.subplot(1,2,2)
plt.stem(freq, np.abs(yf), 'b', markerfmt='bo',  label='right ear')
plt.stem(freq, np.abs(yf2), 'g', markerfmt='go', label='left ear')
plt.xlim(0, 250)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.savefig('entrain_fft_lit.pdf')
# plt.grid()
plt.show()