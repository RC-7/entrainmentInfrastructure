from numpy.fft import fft
import numpy as np
from scipy import signal


class Analyser:
    def __init__(self):
        pass

    @staticmethod
    def generate_fft(data, sampling_rate):
        fft_data = fft(data)
        number_data_points = len(fft_data)
        n = np.arange(number_data_points)
        time_period = number_data_points / sampling_rate
        freq = n / time_period
        return [fft_data, freq]

    @staticmethod
    def butter_highpass(cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(N=order, Wn=normal_cutoff, btype="hp", analog=False, fs=fs)
        return b, a

    @staticmethod
    def butter_bandpass(cutoff_low, cutoff_high, fs, order=4):
        b, a = signal.butter(order, [cutoff_low, cutoff_high], fs=fs, btype='band')
        return b, a

    @staticmethod
    def filter_data(filter, data):
        b = filter[0]
        a = filter[1]
        return signal.filtfilt(b, a, data)
