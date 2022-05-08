from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


ds_name = 'full_run'
filename = f'custom_suite/Full_run/{ds_name}.h5'

t, dt = np.linspace(0, 1, 513, retstep=True)
fs = 1/dt
w = 6.
sig = np.cos(2*np.pi*10*t)
freq = np.linspace(1, 50, 200)
widths = w*fs / (2*freq*np.pi)
cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
plt.show()
