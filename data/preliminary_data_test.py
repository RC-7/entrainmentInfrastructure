import math
import time

from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import signal
from scipy.signal import freqz
from scipy.signal import correlate
from scipy.stats import pearsonr
from math import sqrt, ceil
from numpy.fft import fft
import mne
import matplotlib as mpl
import pandas as pd
from scipy import signal
from scipy.signal import stft
import matplotlib.ticker as plticker
from sklearn.cross_decomposition import CCA
import emd
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from sklearn.metrics import mean_squared_error

ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'F1', 'AF8', 'F7', 'F5', 'F3', 'AF4', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP4', 'CPz', 'CP2', 'CP1', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1', 'A2']

eeg_bands = {'Delta': (0.5, 4),
             'Theta': (4, 8),
             'Alpha': (8, 13),
             'Beta': (13, 30),
             'Gamma': (30, 45)}

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


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(N=order, Wn=normal_cutoff, btype="hp", analog=False, fs=fs)
    return b, a


def butter_bandpass(cutoff_low, cutoff_high, fs, order=4):
    b, a = signal.butter(order, [cutoff_low, cutoff_high], fs=fs, btype='band')
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


def get_events(mne_raw_data):
    return mne.find_events(mne_raw_data)


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
    # electrodes_to_plot = [0, 1, 3, 62, 63]

    # electrodes_to_plot = [1, 62, 63]
    if not same_axis:
        [row, column] = get_subplot_dimensions(electrodes_to_plot)
        fig_size = 1 * len(electrodes_to_plot)
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
            ax.plot(x_val, data_to_plot, label=str(i + 1))
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
    # electrodes_to_plot = [0, 1, 2, 63]
    # electrodes_to_plot = [0, 1, 3]
    row = 0
    column = 0
    active_row = 0
    active_column = 0
    if not same_axis:
        [row, column] = get_subplot_dimensions(electrodes_to_plot)
        fig_size = 1 * len(electrodes_to_plot)
        fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
        fig.tight_layout(pad=0.5)  # edit me when axis labels are added
        fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    else:
        fig, ax = plt.subplots()
        ax.set(xlim=[-0.01, f_lim])
    for i in electrodes_to_plot:
        if built_filter is None:
            data_to_plot = eeg_data[np_slice_indexes[i]]
        else:
            data_to_plot = butter_highpass_filter(abs(eeg_data[np_slice_indexes[i]]), existing_filter=built_filter)
        [fft_data, freq] = get_fft(data_to_plot, SAMPLING_SPEED)
        if not same_axis:
            ax[active_row, active_column].stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b")
            # ax[active_row, active_column].set(xlabel='Freq (Hz)', ylabel='Magnitude', xlim=f_lim)
            ax[active_row, active_column].set(xlim=[-5, f_lim])
            ax[active_row, active_column].set_title(f'Channel {i + 1} ')
            # ax[active_row, active_column].xlim(0, f_lim)
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
        else:
            ax.stem(freq, np.abs(fft_data), 'b', markerfmt=" ", basefmt="-b", label=str(i))
    if not same_axis:
        if active_column != 0:
            for j in range(active_column, column):
                fig.delaxes(ax[active_row, j])
    if not save:
        plt.show()
    else:
        plt.savefig(filename)


def get_binnned_fft(eeg_data):
    [fft_data, freq] = get_fft(eeg_data, SAMPLING_SPEED)
    absolute_fft_values = np.absolute(fft_data)
    sample_frequencies = np.fft.rfftfreq(len(eeg_data), 1.0 / SAMPLING_SPEED)
    # sample_frequencies = freq
    eeg_band_fft = dict()
    for band in eeg_bands:
        band_frequency_values = np.where((sample_frequencies >= eeg_bands[band][0]) &
                                         (sample_frequencies <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.max(absolute_fft_values[band_frequency_values])
    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    # print(df)
    return df


# TODO Add cleaning / logic here
def plot_band_changes(eeg_data_mne, tmin_crop, tmax_crop, electrodes_to_plot, np_slice_indexes, save=False,
                      filename='', only_alpha=False):
    # 10 s FFT windows
    fft_window = 1
    current_window_low = tmin_crop
    max_value = 0
    bands = {}
    active_row = 0
    active_column = 0
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    timesteps = []
    while current_window_low + fft_window <= tmax_crop:
        timesteps.append(current_window_low)
        current_window_low += fft_window
    current_window_low = tmin_crop
    for i in range(62):
        bands[i] = pd.DataFrame(index=timesteps, columns=band_names)
    while current_window_low + fft_window <= tmax_crop:
        fft_data = eeg_data_mne.copy()
        fft_data.crop(tmin=current_window_low, tmax=current_window_low + fft_window).load_data()
        data = fft_data.get_data().transpose()
        # print(len(data))
        for i in electrodes_to_plot:
            data_to_analyse = data[np_slice_indexes[i]]
            df = get_binnned_fft(data_to_analyse)
            values = df['val']
            count = 0
            for b in band_names:
                if b == 'Alpha' and only_alpha:
                    bands[i][b][current_window_low] = values[count]
                elif not only_alpha:
                    bands[i][b][current_window_low] = values[count]
                count += 1
            if df['val'].max() > max_value:
                max_value = df['val'].max()
        current_window_low += fft_window

    [row, column] = get_subplot_dimensions(electrodes_to_plot)
    fig_size = 1 * len(electrodes_to_plot)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added

    for i in electrodes_to_plot:
        df = bands[i]
        my_colors = [(0.50, x / 4.0, x / 5.0) for x in range(len(df))]
        df.plot(xticks=df.index, y=band_names, legend=False, ax=ax[active_row, active_column])
        ax[active_row, active_column].set_title(ch_names[i])

        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    for j in range(active_column, column):
        fig.delaxes(ax[active_row, j])
    if not only_alpha:
        for c in range(column):
            for r in range(row):
                ax[r, c].set_ylim([0, max_value])
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower right')
    if not save:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close(fig)


def plot_fft_binned(eeg_data, electrodes_to_plot, np_slice_indexes, built_filter=None, same_axis=True,
                    save=False, filename=''):
    row = 0
    column = 0
    active_row = 0
    active_column = 0
    max_value = 0
    # electrodes_to_plot = [0, 1, 3]
    if not same_axis:
        [row, column] = get_subplot_dimensions(electrodes_to_plot)
        fig_size = 1 * len(electrodes_to_plot)
        fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
        fig.tight_layout(pad=0.5)  # edit me when axis labels are added
        fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    else:
        fig, ax = plt.subplots()
    for i in electrodes_to_plot:
        if built_filter is None:
            data_to_plot = eeg_data[np_slice_indexes[i]]
        else:
            data_to_plot = butter_highpass_filter(abs(eeg_data[np_slice_indexes[i]]), existing_filter=built_filter)

        df = get_binnned_fft(data_to_plot)

        my_colors = [(0.50, x / 4.0, x / 5.0) for x in range(len(df))]
        if not same_axis:
            if df['val'].max() > max_value:
                max_value = df['val'].max()
            df.plot.bar(x='band', y='val', legend=False, color=my_colors, ax=ax[active_row, active_column])
            ax[active_row, active_column].set_title(f'Channel {i + 1} ')
            # ax[active_row, active_column].set_xlabel("EEG band")
            # ax[active_row, active_column].set_ylabel("Maximum band Amplitude")
            # ax[active_row, active_column].set_ylim([0, 20000])
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
        else:
            ax = df.plot.bar(x='band', y='val', legend=False, color=my_colors)
            ax.set_xlabel("EEG band")
            ax.set_ylabel("Maximum band Amplitude")
    if not same_axis:
        if active_column != 0:
            for j in range(active_column, column):
                fig.delaxes(ax[active_row, j])
        for c in range(column):
            for r in range(row):
                ax[r, c].set_ylim([0, max_value])
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


def generate_mne_raw_with_info(file_type, file_path, patch_data=False, filter_data=False):
    if file_type == 'csv':
        full_eeg_data = get_data_csv(file_path)
        if patch_data:  # Completely breaks data, but needed for testing with current ds
            eeg_data = get_data_from_filter_obscured(full_eeg_data)
        else:
            eeg_data = full_eeg_data  # full_eeg_data[250:500, :] to remove filter data
        if filter_data:
            generated_filter = butter_highpass(0.001, SAMPLING_SPEED, order=5)
            for i in range(64):
                index = np.index_exp[:, i]
                eeg_data[index] = butter_highpass_filter((eeg_data[index]), existing_filter=generated_filter)
    else:
        hf = h5py.File(file_path, 'r')
        if 'hdf5' in file_path:
            tst = hf['RawData']
            tst_samples = tst['Samples']
            eeg_data = tst_samples[()]  # () gets all data
            # print(len(eeg_data))
            # eeg_data = eeg_data[92169-512:92169, :]
            eeg_data = eeg_data[:, :]
            for i in range(64):
                index = np.index_exp[:, i]
                eeg_data[index] = eeg_data[index] - eeg_data[:, 62]
        else:
            samples = hf['raw_data']
            eeg_data = samples[()]
            # eeg_data = eeg_data[100*512:150*512, :]

    # print(len(eeg_data))
    # print(eeg_data.transpose())
    ch_types = ['eeg'] * 64

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=SAMPLING_SPEED)  # TODO flesh out with real cap info
    info.set_montage('standard_1020')  # Will auto set channel names on real cap
    info['description'] = 'My custom dataset'
    raw = mne.io.RawArray(eeg_data.transpose()[0:64], info)
    raw.filter(l_freq=1., h_freq=50)  # removing slow drifts
    # raw.filter(l_freq=1., h_freq=None)  # removing slow drifts
    # raw.filter(l_freq=8, h_freq=30)  # removing slow drifts
    return [raw, info]


def get_fft_mne(data):
    events = mne.find_events(data, stim_channel='STI 014')


def clean_mne_data_ica(raw_data):
    electrodes_to_plot = [x for x in range(64)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]

    # plot_filtered(raw_data.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=True,
    #               filename='sinTest_pre_filter.png')
    # plot_fft(raw_data.get_data().transpose(), electrodes_to_plot, index_dict, f_lim=50, same_axis=False, save=True,
    #          filename='sinTest_pre_filter_fft.png')
    #
    # plot_fft_binned(raw_data.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=True,
    #                 filename=f'pre-filter_EEG_FT_BINNED.png')

    filt_int = raw_data.copy().filter(l_freq=0.01, h_freq=100)
    filt_raw = filt_int.copy().filter(l_freq=48, h_freq=52)

    plot_filtered(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
                  filename='sinTest_post_filter.png')
    plot_fft(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, f_lim=50, same_axis=False, save=False,
             filename='sinTest_post_filter_fft.png')

    plot_fft_binned(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
                    filename=f'post-filter_EEG_FT_BINNED.png')

    # ica = mne.preprocessing.ICA(n_components=12, max_iter='auto', random_state=97)
    # ica.fit(filt_raw)
    # ica.plot_sources(filt_raw, show_scrollbars=True)
    # ica.plot_components()

    # ica.plot_properties(raw_data, picks=[0, 1])

    # removing the components
    # ica.exclude = [2, 7, 2, 10]  # Removing ICA components
    # reconstructed_raw = filt_raw.copy()
    # out = ica.apply(reconstructed_raw)
    # data = out.get_data()

    #
    # # raw_data.plot(show_scrollbars=False)
    # # reconstructed_raw.plot(show_scrollbars=True, start=0, duration=1000)
    # # ica.plot_sources(reconstructed_raw, show_scrollbars=False)
    #

    # print(len(data))
    # print(len(data[0]))

    # eeg_data.transpose()

    # plot_filtered(data.transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
    #               filename='one_min_patch_data_PostBasicICA.png')
    # plot_fft(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, f_lim=50, same_axis=False, save=False,
    #          filename='one_min_patch_data_PostBasicICA_fft.png')
    # plot_fft_binned(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
    #                 filename=f'post_ICA_EEG_FT_BINNED.png')
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


def do_some_hdfs5_analysis(filename, source='custom', filter_data=False, saved_image=''):
    hf = h5py.File(filename, 'r')
    if source == 'custom':
        samples = hf['raw_data']
        eeg_data = samples[()]  # () gets all data
    else:
        tst = hf['RawData']
        tst_samples = tst['Samples']
        eeg_data = tst_samples[()]  # () gets all data
    # electrodes_to_plot = [0, 1, 2, 3, 4, 5, 6]
    print('----------------------------------------------------')
    print('----------- EEG Data dimensions --------------------')
    print(f'Rows: {len(eeg_data)}')
    print(f'Columns: {len(eeg_data[0])}')
    print('----------------------------------------------------')
    electrodes_to_plot = [x for x in range(64)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]
        # index_dict[i] = np.index_exp[90*512:100*512, i]
        # index_dict[i] = np.index_exp[51200:87040, i]
        # index_dict[i] = np.index_exp[200:1000, i] # Swap me if you want to remove filter data
    if filter_data:
        [b, a] = butter_highpass(0.00000001, SAMPLING_SPEED, order=5)
        plot_filtered(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], same_axis=False)
        plot_fft(eeg_data, electrodes_to_plot, index_dict, built_filter=[b, a], f_lim=20, same_axis=False)
    else:
        plot_filtered(eeg_data, electrodes_to_plot, index_dict, same_axis=True, save=False,
                      filename=f'{saved_image}_EEG_Raw.png')
        # plot_fft(eeg_data, electrodes_to_plot, index_dict, f_lim=20, same_axis=False, save=False,
        #          filename=f'{saved_image}_EEG_FT.png')
        # plot_fft_binned(eeg_data, electrodes_to_plot, index_dict, same_axis=False, save=True,
        #                 filename=f'{saved_image}_EEG_FT_BINNED.png')


def view_data(raw_data, tmin_crop=None, tmax_crop=None):
    data_to_view = raw_data.copy()
    if tmax_crop is not None and tmin_crop is not None:
        data_to_view.crop(tmin=tmin_crop, tmax=tmax_crop).load_data()
    elif tmax_crop is None and tmin_crop is not None:
        data_to_view.crop(tmin=tmin_crop).load_data()
    data_to_view.plot(scalings='auto')
    print('--------------------------------------------------')
    print(f'max: {type(data_to_view.get_data().transpose())}')
    print(f'max: {np.amax(data_to_view.get_data().transpose())}')
    print(f'min: {np.amin(data_to_view.get_data().transpose())}')


def test_mne_clean_and_ref(raw_data, tmin_crop, tmax_crop):
    # Not plotting for reference electrodes
    electrodes_to_plot = [x for x in range(62)]
    raw_data.crop(tmin=tmin_crop, tmax=tmax_crop).load_data()
    index_dict = {}
    # raw_data.pick([ch_names[n] for n in range(0, 3)])
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]

    # filt_int = raw_data.copy().filter(l_freq=0.01, h_freq=100)
    # filt_raw = filt_int.copy().filter(l_freq=49.99999, h_freq=50)

    filt_raw = raw_data

    # Filter data
    # filt_raw.set_eeg_reference(ref_channels=['A1'])
    # filt_raw.plot(scalings='auto')

    ica = mne.preprocessing.ICA(n_components=12, max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    # ica.plot_properties(filt_raw)
    # Removing EOG
    # ica.plot_sources(filt_raw, show_scrollbars=True)
    ica.exclude = [0, 4]
    # ica.apply(filt_raw)
    # ica.plot_components()
    ica.apply(filt_raw)
    filt_raw.plot(scalings='auto')
    # filt_raw.plot_psd()
    # filt_raw.plot(scalings='auto')
    # Ref doesn't change output ...
    # raw_bip_ref = mne.set_bipolar_reference(filt_raw, anode=ch_names,
    #                                         cathode=['A1' for i in range(64)])
    # raw_bip_ref.plot(scalings='auto')
    #
    # raw_bip_ref.plot()
    #

    # plot_filtered(raw_bip_ref.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
    #               filename=f'yes_EEG_Raw.png')
    # raw_bip_ref.plot_psd(fmin=2., fmax=40., average=True, spatial_colors=False)

    # plot_fft_binned(filt_raw.get_data().transpose(), electrodes_to_plot, index_dict, same_axis=False, save=False,
    #                 filename=f'e_o_72-76s_BINNED.png')

    # raw_bip_ref.plot(scalings='auto')
    # filt_raw.set_eeg_reference('average', projection=True)
    # filt_raw.plot(scalings='auto')
    return filt_raw


def remove_blinks(raw_data):
    # print(len(raw_data.get_data()[0])/SAMPLING_SPEED)
    initial_end = len(raw_data.get_data()[0]) / SAMPLING_SPEED
    ica_period = 100
    tmin = 100
    tmax = tmin + ica_period
    while tmax < initial_end:
        test_Ds = raw_data.copy()
        test_Ds.crop(tmin=tmin, tmax=tmax).load_data()
        ica = mne.preprocessing.ICA(n_components=12, max_iter='auto', random_state=97, verbose='WARNING')
        ica.fit(test_Ds, verbose='WARNING')
        ica.exclude = [0]
        # ica.plot_sources(test_Ds, show_scrollbars=True)
        ica.apply(test_Ds, verbose='WARNING')
        raw_data.append(test_Ds)
        tmin += ica_period
        if tmax + ica_period > initial_end:
            # print(initial_end)
            tmax = initial_end
        else:
            tmax += ica_period
    raw_data.crop(tmin=initial_end).load_data()
    # print(len(raw_data.get_data()[0])/SAMPLING_SPEED)
    # view_data(raw_data)
    return raw_data


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def stft_test(eeg_data, electrodes_to_plot, np_slice_indexes, save=False, filename=None, plot_averaged=False):
    active_row = 0
    active_column = 0
    [row, column] = get_subplot_dimensions(electrodes_to_plot)
    fig_size = 1 * len(electrodes_to_plot)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    data = eeg_data.get_data().transpose()
    for i in electrodes_to_plot:
        data_to_plot = data[np_slice_indexes[i]]
        samples_per_ft = 100
        overlap = 10
        f, t, Zxx = signal.stft(x=data_to_plot, fs=SAMPLING_SPEED, nperseg=samples_per_ft, noverlap=overlap, nfft=512)

        if plot_averaged:
            abs_power = np.abs(Zxx)
            alpha_average_values = []
            for j in range(len(abs_power[0])):
                alpha_values = []
                for z in range(len(f)):
                    if 22 >= f[z] >= 19:
                        alpha_values.append(abs_power[z, j])
                ave_alpha = np.mean(alpha_values)   # Decide on what to use here
                alpha_average_values.append(ave_alpha)
            time_averaged = []
            window_averaging = 20
            for j in range(0, len(alpha_average_values), window_averaging):
                if j+window_averaging < len(alpha_average_values):
                    end_index = j+window_averaging
                else:
                    end_index = len(alpha_average_values) - 1
                values = alpha_average_values[j:end_index]
                time_averaged.append(np.mean(values))
            ax[active_row, active_column].plot(time_averaged, 'o')
            ma = moving_average(time_averaged, 20)
            ax[active_row, active_column].plot(ma)

            max_lim = np.std(time_averaged) + np.mean(time_averaged)
            min_lim = np.mean(time_averaged) - np.std(time_averaged)
            ax[active_row, active_column].set(ylim=[min_lim, max_lim])
        else:
            ax[active_row, active_column].pcolormesh(t, f[18:21], np.abs(Zxx[18:21]), cmap='viridis', shading='gouraud')
            # ax[active_row, active_column].set(ylim=[10, 30])

        ax[active_row, active_column].set_title(ch_names[i])
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
        # ax[active_row, active_column].title('STFT Magnitude')
        # ax[active_row, active_column].ylabel('Frequency [Hz]')
        # ax[active_row, active_column].xlabel('Time [sec]')
    if not save:
        plt.show()
    else:
        # plt.ioff()
        plt.savefig(filename)
        plt.close(fig)


def alpha_band_stft_test(eeg_data, electrodes_to_plot, np_slice_indexes, save=False, filename=None):
    active_row = 0
    active_column = 0
    [row, column] = get_subplot_dimensions(electrodes_to_plot)
    fig_size = 1 * len(electrodes_to_plot)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    data = eeg_data.get_data().transpose()
    for i in electrodes_to_plot:
        data_to_plot = data[np_slice_indexes[i]]
        # print(data_to_plot)
        samples_per_ft = 1024
        overlap = 100
        f, t, Zxx = signal.stft(x=data_to_plot, fs=SAMPLING_SPEED, nperseg=samples_per_ft, noverlap=overlap)
        abs_power = np.abs(Zxx)
        band_values = []
        # for j in range(len(abs_power)):

        # print(Zxx)
        print(len(abs_power))
        print(len(abs_power[0]))
        print(len(f))
        ax[active_row, active_column].pcolormesh(t, f[11:], np.abs(Zxx[11:]), cmap='viridis', shading='gouraud')  # vmin=0, vmax=amp,
        # ax[active_row, active_column].plot(data_to_plot)
        ax[active_row, active_column].set_title(ch_names[i])
        ax[active_row, active_column].set(ylim=[6, 50])
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
        # ax[active_row, active_column].title('STFT Magnitude')
        # ax[active_row, active_column].ylabel('Frequency [Hz]')
        # ax[active_row, active_column].xlabel('Time [sec]')
    if not save:
        plt.show()
    else:
        # plt.ioff()
        plt.savefig(filename)
        plt.close(fig)

def crop_data(raw_data, tmin_crop=None, tmax_crop=None):
    data = raw_data.copy()
    if tmax_crop is not None and tmin_crop is not None:
        data.crop(tmin=tmin_crop, tmax=tmax_crop).load_data()
    elif tmax_crop is None and tmin_crop is not None:
        data.crop(tmin=tmin_crop).load_data()
    return data


def test_psd(data, electrodes_to_plot, np_slice_indexes):
    row = 0
    column = 0
    active_row = 0
    active_column = 0
    [row, column] = get_subplot_dimensions(electrodes_to_plot)
    fig_size = 1 * len(electrodes_to_plot)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.8)  # edit me when axis labels are added
    for i in range(63):
        # ax[active_row, active_column].psd(data[np_slice_indexes[i]], 512, 1/512)
        win = 3 * SAMPLING_SPEED
        freqs, psd = signal.welch(data[np_slice_indexes[i]], SAMPLING_SPEED, nperseg=win)
        ax[active_row, active_column].plot(freqs, psd, color='k', lw=2)
        ax[active_row, active_column].set(xlim=[-5, 20])
        ax[active_row, active_column].set_title(f'Channel {ch_names[i]} ')
        loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        ax[active_row, active_column].xaxis.set_major_locator(loc)
        ax[active_row, active_column].grid()
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    filename = 'testpsd_280-284s_zoomed'
    # plt.grid()
    plt.savefig(filename)


def plot_blinks(raw_data, window):
    current_window_low = 0
    max_sample = len(raw_data[:, 0])
    output = []
    bool_of_blink = []
    eb_indices = []
    eb_templates = []
    eb_template_index = []
    blink_found = False
    while current_window_low + window <= max_sample:
        index_fp1 = np.index_exp[current_window_low:current_window_low + window, ch_names.index('Fp1')]
        index_fp2 = np.index_exp[current_window_low:current_window_low + window, ch_names.index('Fp2')]
        correl = pearsonr(raw_data[index_fp1], raw_data[index_fp2])
        # ID epochs with blinks based on Fp1 and Fp2 correlation
        if correl[0] > 0.9:
            mean_value_fp1 = np.mean(raw_data[index_fp1])
            std_div_fp1 = np.std(raw_data[index_fp1])
            f = lambda x: np.abs(x - mean_value_fp1)
            displacement = f(raw_data[index_fp1])
            for i in range(len(displacement)):
                if np.max(displacement) > mean_value_fp1 + 3 * std_div_fp1:
                    eb_start = i - 200
                    eb_index_low = current_window_low + eb_start
                    eb_index_high = eb_index_low + 512
                    eb_indices.append([eb_index_low, eb_index_high])
                    index_eb = np.index_exp[eb_index_low:eb_index_high, ch_names.index('Fp1')]
                    eb_templates.append(raw_data[index_eb])
                    blink_found = True
                    break
        if not blink_found:
            current_window_low += window
        else:
            current_window_low = eb_index_high + 200
            blink_found = False
    blink_index = 0
    min_index = eb_indices[0][0]
    max_index = eb_indices[0][1]
    for i in range(len(raw_data[:, 0])):
        if min_index <= i < max_index:
            bool_of_blink.append(50)
        elif i == max_index:
            bool_of_blink.append(0)
            if blink_index != len(eb_indices) - 1:
                blink_index += 1
                min_index = eb_indices[blink_index][0]
                max_index = eb_indices[blink_index][1]
        else:
            bool_of_blink.append(0)

    temp = np.vstack([ele for ele in [np.transpose(raw_data), bool_of_blink]])
    raw_data = np.transpose(temp)
    electrodes_to_plot = [0, 1, 3, 64]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]
    plot_filtered(raw_data, electrodes_to_plot, index_dict, same_axis=True, save=False,
                  filename=f'yes_EEG_Raw.png')
    plt.show()


def get_eog_template_raw(raw_data, window):
    current_window_low = 0
    max_sample = len(raw_data[:, 0])
    eb_indices = []
    eb_templates = []
    eb_template_index = []
    blink_found = False
    while current_window_low + window <= max_sample:
        index_fp1 = np.index_exp[current_window_low:current_window_low + window, ch_names.index('Fp1')]
        index_fp2 = np.index_exp[current_window_low:current_window_low + window, ch_names.index('Fp2')]
        correl = pearsonr(raw_data[index_fp1], raw_data[index_fp2])
        # ID epochs with blinks based on Fp1 and Fp2 correlation
        if correl[0] > 0.9:
            mean_value_fp1 = np.mean(raw_data[index_fp1])
            std_div_fp1 = np.std(raw_data[index_fp1])
            f = lambda x: np.abs(x - mean_value_fp1)
            displacement = f(raw_data[index_fp1])
            for i in range(len(displacement)):
                if displacement[i] > mean_value_fp1 + 2 * std_div_fp1:
                    eb_start = i - 200
                    eb_index_low = current_window_low + eb_start
                    eb_index_high = eb_index_low + SAMPLING_SPEED
                    eb_indices.append([eb_index_low, eb_index_high])
                    index_eb = np.index_exp[eb_index_low:eb_index_high, ch_names.index('Fp1')]
                    eb_templates.append(raw_data[index_eb])
                    blink_found = True
                    if len(eb_templates) >= 2 and len(eb_template_index) < 2:
                        for j in range(len(eb_templates)):
                            for z in range(len(eb_templates)):
                                if j == z:
                                    continue
                                if len(eb_templates[z]) != len(eb_templates[j]):
                                    continue
                                if len(eb_template_index) >= 2:
                                    break
                                correl_templates = pearsonr(eb_templates[j], eb_templates[z])
                                if correl_templates[0] > 0.9:
                                    return [eb_templates[j], eb_templates[z]]

                    break
        if not blink_found:
            current_window_low += window
        else:
            current_window_low = eb_index_high + 200
            blink_found = False
    # ave_correl = []
    # if len(eb_templates) >= 2 and len(eb_template_index) < 2:
    #     for j in range(len(eb_templates)):
    #         correl_values = []
    #         for z in range(len(eb_templates)):
    #             if j == z:
    #                 continue
    #             if len(eb_templates[z]) != len(eb_templates[j]):
    #                 continue
    #             if len(eb_template_index) >= 2:
    #                 break
    #             correl_templates = pearsonr(eb_templates[j], eb_templates[z])
    #             correl_values.append(correl_templates[0])
    #         ave_correl.append([correl_values])
    # tst = []
    # for i in range(len(ave_correl)):
    #     tst.append(np.mean(np.abs(ave_correl[i])))
    # max_correl = np.argmax(tst)
    # second_max = np.argmax(ave_correl[max_correl])
    # return [eb_templates[max_correl], eb_templates[second_max]]


def get_sd(previous_iteration, current_iteration):
    sd = 0
    for i in range(len(previous_iteration)):
        sd += ((previous_iteration[i] - current_iteration[i]) ** 2) / ((previous_iteration[i]) ** 2)
    return sd


def get_next_imf(x, sd_thresh=0.2):
    sifting_signal = x.copy()
    continue_sift = True

    while continue_sift:
        upper_env = emd.sift.interp_envelope(sifting_signal, mode='upper')
        lower_env = emd.sift.interp_envelope(sifting_signal, mode='lower')
        avg_env = (upper_env + lower_env) / 2

        tmp = sifting_signal - avg_env

        sd = get_sd(sifting_signal, tmp)

        stop = sd <= sd_thresh

        sifting_signal = sifting_signal - avg_env
        if stop:
            continue_sift = False

    return sifting_signal


def get_eog_template_emd(raw_template_data):
    emd_templates = []
    # Can use EMD sift function below
    # emd_templates = emd.sift.sift(raw_template_data[1], max_imfs=5, imf_opts={'sd_thresh': 0.2})
    # emd.plotting.plot_imfs(emd_templates, cmap=True, scale_y=True)
    for i in range(len(raw_template_data)):
        imf_templates = []
        imf = raw_template_data[i].copy()
        for j in range(5):
            imf_iter = get_next_imf(imf)
            imf = imf - imf_iter
            imf_templates.append(imf_iter)
            imf_templates.append(imf)
        template = imf_templates[2] + imf_templates[3] + imf_templates[4] + imf_templates[5]
        emd_templates.append(template)

    template = emd_templates[0] + emd_templates[1] / 2

    return template


def cross_correlate(signal1, signal2):
    numerator = 0
    s1_sq_sum = 0
    s2_sq_sum = 0

    for i in range(len(signal1)):
        numerator += signal1[i] * signal2[i]
        s1_sq_sum += signal1[i] ** 2
        s2_sq_sum += signal2[i] ** 2
    denominator = math.sqrt(s1_sq_sum) * math.sqrt(s2_sq_sum)

    cross_correlation_coefficient = numerator / denominator
    return cross_correlation_coefficient


def save_hdfs5(filename, data):
    hf = h5py.File(filename, 'a')
    dataset_name = 'raw_data'
    hf.create_dataset(dataset_name, data=data, chunks=True, maxshape=(None, len(data[0])))
    hf.close()


def remove_blinks_cca(raw_data, blink_template, testing=False):
    window = len(blink_template)
    print(window)
    max_sample = len(raw_data[:, 0])
    print(max_sample)
    current_window_low = 0
    sliding_iter = 50  # Edit me for optimisation
    blink_found = False
    blink_counter = 0
    tic = time.perf_counter()
    while current_window_low + window <= max_sample:
        index_electrode = np.index_exp[current_window_low:current_window_low + window, 0]
        if len(blink_template) != len(raw_data[index_electrode]):
            print('Blink template length mismatch')
            current_window_low += window
            continue
        correl = cross_correlate(raw_data[index_electrode], blink_template)

        if correl > 0.5:
            mean_value_fp1 = np.mean(raw_data[index_electrode])
            std_div_fp1 = np.std(raw_data[index_electrode])
            f = lambda x: np.abs(x - mean_value_fp1)
            displacement = f(raw_data[index_electrode])
            if np.max(displacement) > mean_value_fp1 + 3 * std_div_fp1:
                index_original = np.index_exp[current_window_low:current_window_low + window - 1, 0:63]
                index_shifted = np.index_exp[current_window_low + 1:current_window_low + window, 0:63]
                # Can play with tolerance to optimise for time
                cca = CCA(n_components=10, scale=False, tol=1E-05, max_iter=500)
                left_window = raw_data[index_original]
                right_window = raw_data[index_shifted]
                cca.fit(left_window, right_window)
                x_c, y_c = cca.transform(left_window, right_window)
                index_test = np.index_exp[:, 0]
                x_c[index_test] = 0  # U matrix
                blink_removed = cca.inverse_transform(x_c)  # Do something with me!
                raw_data[index_original] = blink_removed
                raw_data = np.delete(raw_data, current_window_low + window, 0)
                blink_counter += 1
                blink_found = True
        if not blink_found:
            current_window_low += sliding_iter
        else:
            current_window_low += sliding_iter
            blink_found = False
    toc = time.perf_counter()
    elapsed = toc - tic
    print(f'Blinks cleaned in data: {blink_counter}')
    print(f"Cleaning {len(raw_data) / 512}s of data took {elapsed:0.4f} seconds")
    electrodes_to_plot = [x for x in range(62)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]
    if testing:
        plot_filtered(raw_data, electrodes_to_plot, index_dict, same_axis=False, save=True,
                      filename=f'removed_blinks.png')
    return raw_data


#     First pass will work with numpy array, not mne data
def clean_cca(raw_data, output_filename):
    window = SAMPLING_SPEED * 2
    known_blink_data = crop_data(raw_data, 390, 450).get_data().transpose()
    template_data = get_eog_template_raw(known_blink_data, window)
    known_blink_data = None
    # fig, axs = plt.subplots(2)
    # axs[0].plot(template_data[0])
    # axs[1].plot(template_data[1])
    blink_template = get_eog_template_emd(template_data)
    # plt.subplots()
    # plt.plot(blink_template)
    # plt.show()
    data_to_clean = crop_data(raw_data, 100).get_data().transpose()
    # plot_blinks(data_to_clean, window)
    cleaned_data = remove_blinks_cca(data_to_clean, blink_template, False)
    save_hdfs5(output_filename, cleaned_data)


def morlet_tf(eeg_data, electrodes_to_plot, np_slice_indexes, save=False, filename=None):
    active_row = 0
    active_column = 0
    [row, column] = get_subplot_dimensions(electrodes_to_plot)
    fig_size = 2 * len(electrodes_to_plot)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    # fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=2)  # edit me when axis labels are added
    data = eeg_data.get_data().transpose()
    for i in electrodes_to_plot:
        data_to_plot = data[np_slice_indexes[i]]
        data_to_plot = data_to_plot[::2]
        downsampling = SAMPLING_SPEED / 2
        w = 6.
        # freq = np.linspace(1, 50, 200)
        freq = np.geomspace(5, 14, 5)
        widths = w * downsampling / (2 * freq * np.pi)
        cwtm = signal.cwt(data_to_plot, signal.morlet2, widths, w=w)
        # print(len(np.abs(cwtm[:, ::100])))
        # print(cwtm[:, ::100])
        magnitudes = np.abs(cwtm[:, ::100])  # Maybe average instead ...
        t = np.linspace(1, len(data_to_plot) / downsampling, len(magnitudes[0]))
        pc = ax[active_row, active_column].pcolormesh(t, freq, magnitudes, cmap='viridis', shading='gouraud')
        plt.colorbar(pc, ax=ax[active_row, active_column])
        # ax[active_row, active_column].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')  # vmin=0, vmax=amp,
        # ax[active_row, active_column].plot(data_to_plot)
        ax[active_row, active_column].set_title(ch_names[i])
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
        # ax[active_row, active_column].title('STFT Magnitude')
        # ax[active_row, active_column].ylabel('Frequency [Hz]')
        # ax[active_row, active_column].xlabel('Time [sec]')
    if not save:
        plt.show()
    else:
        # plt.ioff()
        plt.savefig(filename)
        plt.close(fig)


def main():
    # do_some_csv_analysis(patch=True)
    # filename = 'gtec/run_3.hdf5'
    ds_name = 'audio_eyes_closed'
    # # ds_name = 'eyes_closed_with_oculus'
    # filename = f'custom_suite/Full_run/{ds_name}.h5'
    filename = f'custom_suite/Full_run_J/{ds_name}.h5'
    output_filename = f'custom_suite/Full_run/{ds_name}_cleaned_V1.h5'
    # do_some_hdfs5_analysis(filename, source='custom', saved_image=ds_name)

    # ds_name = 'H_e_c'
    # ds_name = 'H_e_blink'
    # filename = f'gtec/{ds_name}.hdf5'
    # do_some_hdfs5_analysis(filename, source='gtec', saved_image=ds_name)
    # file_type = 'hdfs5'
    # file_path = 'gtec/run_3.hdf5'

    #################################
    ############## mne ##############
    #################################
    file_type = 'hdfs'
    # # file_path = 'custom_suite/one_minute_half_fixed.csv'
    # # file_path = 'testData/sinTest.csv'
    # electrodes_to_plot = [0, 1, 2, 3, 4, 63]
    [raw, info] = generate_mne_raw_with_info(file_type, filename, patch_data=False, filter_data=False)
    # view_data(raw)
    #
    # tmin_crop = 100
    # tmax_crop = 175

    # ica_data = remove_blinks(raw)
    electrodes_to_plot = [x for x in range(62)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]
    tmin_crop = 500
    tmax_crop = 550
    # # tmax_crop = 130
    cropped_data = crop_data(raw, 40)
    # view_data(cropped_data)
    # morlet_tf(cropped_data, electrodes_to_plot, index_dict, save=True,
    #           filename='morlet_test_500-550s_collapsed_alpha.png')

    # plot_band_changes(cropped_data, tmin_crop, tmax_crop, electrodes_to_plot, index_dict, only_alpha=True, save=True,
    #                   filename='test_band_changes.png')
    # clean_cca(raw, output_filename)
    # data = cropped_data.get_data().transpose()
    # test_psd(data, electrodes_to_plot, index_dict)

    # raw_ica_removed = test_mne_clean_and_ref(raw, tmin_crop, tmax_crop)

    # # # clean_mne_data_ica(raw)
    # tmin_crop = 0
    # tmax_crop = len(raw_ica_removed.get_data()[0])/512
    # index_dict = {}
    # # raw_data.pick([ch_names[n] for n in range(0, 3)])
    stft_test(cropped_data, electrodes_to_plot, index_dict, save=True, filename='J_alpha_eyes_closed_19-22.png', plot_averaged=True)
    # stft_test(raw, electrodes_to_plot, index_dict, save=True, filename='me_test.png')
    # plot_band_changes(raw_ica_removed, tmin_crop, tmax_crop, electrodes_to_plot, index_dict, only_alpha=True, save=True,
    #                   filename='entrain_1000-1379s_ICA_removed_alpha_max_1s_fft.png')
    # plot_sensor_locations(raw)
    # plot_topo_map(raw)


if __name__ == '__main__':
    main()
