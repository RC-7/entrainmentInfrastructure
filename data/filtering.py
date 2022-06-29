from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np

from constants import SAMPLING_SPEED


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(N=order, Wn=normal_cutoff, btype="hp", analog=False, fs=fs)
    return b, a


def butter_bandpass(cutoff_low, cutoff_high, fs, order=4):
    b, a = signal.butter(order, [cutoff_low, cutoff_high], fs=fs, btype='band')
    return b, a


def butter_highpass_filter(data, cutoff=0.001, fs=SAMPLING_SPEED, existing_filter=None, order=5):
    if existing_filter is None:
        b, a = butter_highpass(cutoff, fs, order=order)
    else:
        b = existing_filter[0]
        a = existing_filter[1]
    y = signal.filtfilt(b, a, data)
    return y


# Plots the frequency response of the filter built
def view_filter(b, a, fs):
    w, h = freqz(b, a, worN=50)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()
    plt.close()
