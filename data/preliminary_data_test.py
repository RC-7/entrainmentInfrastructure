from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import signal
from scipy.signal import freqz
from math import sqrt, ceil
from numpy.fft import fft

plt.autoscale(True)


def get_data_csv(filename):
    return genfromtxt(filename, delimiter=',')


def get_fft(data, sampling_rate):
    fft_data = fft(data)
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


def butter_highpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = 5 / nyq
    b, a = signal.butter(N=order, Wn=normal_cutoff, btype="hp", analog=False, fs=fs)
    return b, a


# Plots the frequency response of the filter built
def view_filter(b, a, fs):
    w, h = freqz(b, a, worN=50)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()
    plt.close()


def butter_highpass_filter(data, cutoff=0.001, fs=521, existing_filter=None, order=5):
    if existing_filter is None:
        b, a = butter_highpass(cutoff, fs, order=order)
    else:
        b = existing_filter[0]
        a = existing_filter[1]
    y = signal.filtfilt(b, a, data)
    return y


def get_subplot_dimentions(electrodes_to_plot):
    row = 0
    column = 0
    if np.abs((len(electrodes_to_plot) / 2) - 2) > 2:
        sqrt_plots = sqrt(len(electrodes_to_plot))
        row = ceil(sqrt_plots)
        column = ceil(sqrt_plots)
    else:
        row = 2
        column = ceil(len(electrodes_to_plot) / 2)
    return [row, column]

def plot_filtered(eeg_data, electrodes_to_plot, np_slice_indexes, same_axis=True, built_filter=None,
                  save=False, filename=''):
    x_val = create_x_values(eeg_data[np_slice_indexes[0]])
    row = 0
    column = 0
    active_row = 0
    active_column = 0
    if not same_axis:
        [row, column] = get_subplot_dimentions(electrodes_to_plot)
        fig, ax = plt.subplots(row, column)
    else:
        fig, ax = plt.subplots()
    for i in electrodes_to_plot:
        # print((eeg_data[250:521, 1]))
        if built_filter is None:
            data_to_plot = eeg_data[np_slice_indexes[i]]
        else:
            data_to_plot = butter_highpass_filter(abs(eeg_data[np_slice_indexes[i]]), existing_filter=built_filter)
        if not same_axis:
            ax[active_row, active_column].plot(x_val, data_to_plot, label=str(i))
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
        else:
            plt.plot(x_val, data_to_plot, label=str(i))
    if same_axis:
        ax.legend()
    else:
        if active_column != 0:
            for j in range(active_column, column):
                fig.delaxes(ax[active_row, j])
    if not save:
        plt.show()
    else:
        plt.savefig(filename)


def plot_fft(eeg_data, electrodes_to_plot, np_slice_indexes, f_lim, built_filter, same_axis=True,
             save=False, filename=''):
    row = 0
    column = 0
    active_row = 0
    active_column = 0
    if not same_axis:
        [row, column] = get_subplot_dimentions(electrodes_to_plot)
        fig, ax = plt.subplots(row, column)
        fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    for i in electrodes_to_plot:
        data_to_plot = butter_highpass_filter(abs(eeg_data[np_slice_indexes[i]]), existing_filter=built_filter)
        [fft_data, freq] = get_fft(data_to_plot, 521)
        print(np.abs(fft_data).max())
        if not same_axis:
            ax[active_row, active_column].stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
            # ax[active_row, active_column].set(xlabel='Freq (Hz)', ylabel='Magnitude', xlim=f_lim)
            ax[active_row, active_column].set(xlim=[0, f_lim])
            ax[active_row, active_column].set_title(f'Channel {i} ')
            # ax[active_row, active_column].xlim(0, f_lim)
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
    if not same_axis:
        if active_column != 0:
            for j in range(active_column, column):
                fig.delaxes(ax[active_row, j])
    if not save:
        plt.show()
    else:
        plt.savefig(filename)


def do_some_csv_analysis():
    eeg_data = get_data_csv('custom_suite/test_short_email_fixed.csv')
    electrodes_to_plot = [0, 3, 20, 22, 30, 32]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[250:500, i]

    [b, a] = butter_highpass(0.00000001, 521, order=5)
    plot_filtered(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], same_axis=False)

    plot_fft(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], f_lim=50, same_axis=False)


def do_some_hdfs5_analysis():
    filename = 'gtec/run_3.hdf5'
    hf = h5py.File(filename, 'r')
    tst = hf['RawData']
    tst_samples = tst['Samples']
    eeg_data = tst_samples[()]  # () gets all data
    electrodes_to_plot = [0, 1, 2, 3, 4, 5, 6]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[250:1000, i]

    [b, a] = butter_highpass(0.00000001, 521, order=5)

    plot_filtered(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], same_axis=False)

    plot_fft(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], f_lim=50, same_axis=False)


def main():
    do_some_csv_analysis()
    # do_some_hdfs5_analysis()


if __name__ == '__main__':
    main()
