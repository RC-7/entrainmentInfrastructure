from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import signal

from numpy.fft import fft


def get_data_csv(filename):
    return genfromtxt(filename, delimiter=',')


def get_fft(data, sampling_rate):
    fft_data = fft(data)
    # plt.savefig('tstcvs_test_bas')
    N = len(fft_data)
    n = np.arange(N)
    T = N / sampling_rate
    freq = n / T
    return [fft_data, freq]


def create_x_values(data, sampling_speed=521):
    diff = 1 / sampling_speed
    x_val = []
    for i in range(len(data)):
        x_val.append(i * diff)
    return x_val


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(N=order, Wn=normal_cutoff, btype="high", analog=False, fs=fs)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# TODO MASSIVE overhaul
def do_some_csv_analysis():
    eeg_data = get_data_csv('custom_suite/test_short_email_fixed.csv')
    # print(eeg_data[:, 0])

    data_plotting = eeg_data[0:200, 0]
    diff = 1 / 521

    x_val = create_x_values(data_plotting)

    fig, ax = plt.subplots()

    electrodes_to_plot = [0, 3, 20, 22, 30, 32]
    for i in electrodes_to_plot:
        plt.plot(x_val, eeg_data[0:200, i], label=str(i))

    # for i in range(20, 22):
    #     plt.plot(x_val, eeg_data[250:500, i], label=str(i))
    #
    # for i in range(30, 32):
    #     plt.plot(x_val, eeg_data[250:500, i], label=str(i))
    ax.legend()

    plt.show()
    #
    # [fft_data, freq] = get_fft(eeg_data[250:500, 0], 521)
    #
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    #
    # plt.stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('FFT Amplitude channel 1 ')
    # plt.xlim(0, 35)
    # [fft_data, freq] = get_fft(eeg_data[250:500, 15], 521)
    # plt.subplot(122)
    # plt.stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('FFT Amplitude channel 15 ')
    # plt.xlim(0, 35)
    # plt.show()


def do_some_hdfs5_analysis():
    filename = 'gtec/run_3.hdf5'
    hf = h5py.File(filename, 'r')
    # data = hf.get('dataset_name').value  # `data` is now an ndarray.
    tst = hf['RawData']
    tst_samples = tst['Samples']
    eeg_data = tst_samples[()]  # () gets all data
    x_val = create_x_values(eeg_data[250:521, :])
    electrodes_to_plot = [1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots()
    for i in electrodes_to_plot:
        plt.plot(x_val, eeg_data[250:521, i], label=str(i))
    ax.legend()
    plt.show()
    print(len(x_val))


def main():
    # do_some_csv_analysis()
    do_some_hdfs5_analysis()


if __name__ == '__main__':
    main()
