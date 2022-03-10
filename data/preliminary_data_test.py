from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import signal
from scipy.signal import freqz
from math import sqrt, ceil
from numpy.fft import fft
import mne
import matplotlib as mpl

# plt.autoscale(True)
SAMPLING_SPEED = 512


def get_data_csv(filename):
    return genfromtxt(filename, delimiter=',')


def get_fft(data, sampling_rate):
    fft_data = fft(data)
    N = len(fft_data)
    n = np.arange(N)
    T = N / sampling_rate
    freq = n / T
    return [fft_data, freq]


def create_x_values(data, sampling_speed=SAMPLING_SPEED):
    diff = 1 / sampling_speed
    x_val = []
    for i in range(len(data)):
        x_val.append(i * diff)
    return x_val


def butter_highpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
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


def butter_highpass_filter(data, cutoff=0.001, fs=SAMPLING_SPEED, existing_filter=None, order=5):
    if existing_filter is None:
        b, a = butter_highpass(cutoff, fs, order=order)
    else:
        b = existing_filter[0]
        a = existing_filter[1]
    y = signal.filtfilt(b, a, data)
    return y


def get_subplot_dimensions(electrodes_to_plot):
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
        [row, column] = get_subplot_dimensions(electrodes_to_plot)
        fig_size = 0.5 * len(electrodes_to_plot)
        fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
        fig.tight_layout(pad=0.8)  # edit me when axis labels are added
    else:
        fig, ax = plt.subplots()
    for i in electrodes_to_plot:
        # print((eeg_data[250:SAMPLING_SPEED, 1]))
        if built_filter is None:
            data_to_plot = eeg_data[np_slice_indexes[i]]
        else:
            data_to_plot = butter_highpass_filter(abs(eeg_data[np_slice_indexes[i]]), existing_filter=built_filter)
        if not same_axis:
            ax[active_row, active_column].plot(x_val, data_to_plot, label=str(i))
            ax[active_row, active_column].set_title(f'Channel {i + 1} ')
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
        else:
            ax.plot(x_val, data_to_plot, label=str(i))
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


def plot_fft(eeg_data, electrodes_to_plot, np_slice_indexes, f_lim, built_filter=None, same_axis=True,
             save=False, filename=''):
    row = 0
    column = 0
    active_row = 0
    active_column = 0
    if not same_axis:
        [row, column] = get_subplot_dimensions(electrodes_to_plot)
        fig_size = 0.5 * len(electrodes_to_plot)
        fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
        fig.tight_layout(pad=0.5)  # edit me when axis labels are added
        fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    for i in electrodes_to_plot:
        if built_filter is None:
            data_to_plot = eeg_data[np_slice_indexes[i]]
        else:
            data_to_plot = butter_highpass_filter(abs(eeg_data[np_slice_indexes[i]]), existing_filter=built_filter)
        [fft_data, freq] = get_fft(data_to_plot, SAMPLING_SPEED)
        if not same_axis:
            ax[active_row, active_column].stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
            # ax[active_row, active_column].set(xlabel='Freq (Hz)', ylabel='Magnitude', xlim=f_lim)
            ax[active_row, active_column].set(xlim=[0, f_lim])
            ax[active_row, active_column].set_title(f'Channel {i + 1} ')
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


def get_data_from_filter_obscured(full_eeg_data):
    index_low = 771
    index_high = 1021
    scoped_data = full_eeg_data[250:SAMPLING_SPEED, :]
    while index_high < len(full_eeg_data):
        index = np.index_exp[index_low:index_high, :]
        scoped_data = np.append(scoped_data, full_eeg_data[index], axis=0)
        index_low += 521
        index_high += 521
    return scoped_data


def generate_mne_raw_with_info(file_type, electrodes_to_plot, file_path, patch_data=False, filter_data=False):
    if file_type == 'csv':
        full_eeg_data = get_data_csv(file_path)
        if patch_data:
            eeg_data = get_data_from_filter_obscured(full_eeg_data)
        else:
            eeg_data = full_eeg_data[250:500, :]
        if filter_data:
            generated_filter = butter_highpass(0.001, SAMPLING_SPEED, order=5)
            for i in range(64):
                index = np.index_exp[:, i]
                eeg_data[index] = butter_highpass_filter((eeg_data[index]), existing_filter=generated_filter)
    else:
        filename = 'gtec/run_3.hdf5'
        hf = h5py.File(filename, 'r')
        tst = hf['RawData']
        tst_samples = tst['Samples']
        eeg_data = tst_samples[()]  # () gets all data
    # print(len(eeg_data))
    # print(eeg_data.transpose())
    ch_types = ['eeg'] * 64

    ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4',
                'O2']

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=SAMPLING_SPEED)  # TODO flesh out with real cap info
    info.set_montage('standard_1020')  # Will auto set channel names on real cap
    info['description'] = 'My custom dataset'
    raw = mne.io.RawArray(eeg_data.transpose()[0:64], info)
    return [raw, info]


def get_fft_mne(data):
    events = mne.find_events(data, stim_channel='STI 014')


def clean_mne_data_ica(raw_data):
    filt_raw = raw_data.copy().filter(l_freq=0.01, h_freq=50)

    electrodes_to_plot = [x for x in range(64)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]

    plot_filtered(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
                        filename='one_min_patch_data_PostFilter.png')

    ica = mne.preprocessing.ICA(n_components=25, max_iter='auto', random_state=97)
    data = ica.fit(filt_raw)
    # ica.plot_sources(raw_data, show_scrollbars=True)
    # ica.plot_components()
    # ica.plot_properties(raw_data, picks=[0, 1])
    ica.exclude = [0, 7]  # Removing ICA components
    # removing the components
    reconstructed_raw = raw_data.copy()
    out = ica.apply(reconstructed_raw)
    # TODO plot the reconstructed data

    # raw_data.plot(show_scrollbars=False)
    # reconstructed_raw.plot(show_scrollbars=True, start=0, duration=1000)
    # ica.plot_sources(reconstructed_raw, show_scrollbars=False)

    data = out.get_data()
    # print(len(data))
    # print(len(data[0]))

    # eeg_data.transpose()

    plot_filtered(data.transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
                  filename='one_min_patch_data_PostBasicICA.png')
    # ica.plot_properties(raw_data, picks=[0, 1])
    # print('here')
    # ica.plot_sources(raw_data, show_scrollbars=True)
    # ica.plot_sources(reconstructed_raw, show_scrollbars=False)
    # plt.show()


# TODO fix colour map and add reference uV values to it's description
def plot_topo_map(raw_data):
    # Allows for expansion to show time data with map on one figure
    times = np.arange(0.05, 0.151, 0.02)
    fig, ax = plt.subplots(ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)
    # It is possible to set own thresholds here
    [a, b] = mne.viz.plot_topomap(raw_data.get_data()[:, 0], raw_data.info, axes=ax,
                                  show=False, sensors=True, ch_type='eeg')
    cmap = a.cmap
    bounds = b.levels
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.set_title('Topographical map of data', fontweight='bold')
    cax = fig.add_axes([0.85, 0.031, 0.03, 0.8])  # fix location
    # cax = plt.axes([0.85, 0.031, 0.03, 0.8])  # fix location
    plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax,
        boundaries=[-150] + bounds + [120],  # Adding values for extensions.
        extend='neither',
        ticks=bounds,
        # spacing='proportional',
        # orientation='horizontal',
        # label='Discrete intervals, some other units',
    )
    plt.show()


def plot_sensor_locations(raw_data):
    fig, ax = plt.subplots(ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)

    # we plot the channel positions with default sphere - the mne way
    raw_data.plot_sensors(axes=ax, show=False)
    ax.set_title('Channel projection', fontweight='bold')
    plt.show()


def do_some_csv_analysis(patch=False):
    full_eeg_data = get_data_csv('custom_suite/one_minute_half_fixed.csv')
    # electrodes_to_plot = [0, 3, 20, 22, 30, 32]
    electrodes_to_plot = [x for x in range(64)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]
    if patch:
        eeg_data = get_data_from_filter_obscured(full_eeg_data)
    else:
        eeg_data = full_eeg_data
    [b, a] = butter_highpass(0.00000001, SAMPLING_SPEED, order=5)
    # plot_filtered(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], same_axis=False)
    # Hardware filtering does this for us!
    plot_filtered(eeg_data, electrodes_to_plot, index_dict, same_axis=False, save=True,
                  filename='one_min_patch_data.png')

    plot_fft(eeg_data, electrodes_to_plot, index_dict, f_lim=50, same_axis=False, save=True,
             filename='one_min_patch_fft.png')


def do_some_hdfs5_analysis(filter_data=False):
    filename = 'gtec/run_3.hdf5'
    hf = h5py.File(filename, 'r')
    tst = hf['RawData']
    tst_samples = tst['Samples']
    eeg_data = tst_samples[()]  # () gets all data
    electrodes_to_plot = [0, 1, 2, 3, 4, 5, 6]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[200:1000, i]
    if filter_data:
        [b, a] = butter_highpass(0.00000001, SAMPLING_SPEED, order=5)
        plot_filtered(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], same_axis=False)
    else:
        plot_filtered(eeg_data, electrodes_to_plot, index_dict, same_axis=False)
    plot_fft(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], f_lim=20, same_axis=False)


def main():
    # do_some_csv_analysis(patch=True)

    # do_some_hdfs5_analysis()
    # file_type = 'hdfs5'
    # file_path = 'gtec/run_3.hdf5'

    #################################
    ############## mne ##############
    #################################
    file_type = 'csv'
    file_path = 'custom_suite/one_minute_half_fixed.csv'
    electrodes_to_plot = [0, 1, 2, 3, 4, 5, 6]
    [raw, info] = generate_mne_raw_with_info(file_type, electrodes_to_plot, file_path,
                                             patch_data=True, filter_data=False)

    clean_mne_data_ica(raw)

    # plot_sensor_locations(raw)
    # plot_topo_map(raw)


if __name__ == '__main__':
    main()
