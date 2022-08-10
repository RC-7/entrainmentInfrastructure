import math

from constants import ch_names, eeg_bands, SAMPLING_SPEED, ch_hemisphere
from scipy import signal
import numpy as np
from util import get_subplot_dimensions, setup_figure, moving_average, figure_handling
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from numpy.fft import fft
from filtering import butter_highpass_filter
import pandas as pd
from scipy.stats import mode



def morlet_tf_region_averged(eeg_data, electrodes_to_plot, np_slice_indexes, save=False, filename=None):
    active_row = 0
    active_column = 0
    ma_global = []
    ma_modal_global = []
    ma_std_global = []
    ma_seventyth_global = []
    data = eeg_data.get_data().transpose()
    for i in electrodes_to_plot:
        data_to_plot = data[np_slice_indexes[i]]
        # data_to_plot = data_to_plot[::2]
        # downsampling = SAMPLING_SPEED / 2
        w = 6.
        # freq = np.linspace(1, 50, 200)
        # freq = np.geomspace(5, 14, 5)
        freq = np.linspace(23, 24, 2)
        widths = w * SAMPLING_SPEED / (2 * freq * np.pi)
        cwtm = signal.cwt(data_to_plot, signal.morlet2, widths, w=w)
        # print(len(np.abs(cwtm[:, ::100])))
        # print(cwtm[:, ::100])
        magnitudes = np.abs(cwtm[:, ::100])  # Maybe average instead ...
        t = np.linspace(1, len(data_to_plot), len(magnitudes[0]))
        averaged_band = np.mean(magnitudes, axis=0)
        time_averaged, time_max, time_min, one_std, modal_values, seventyth_percent = extract_averages(averaged_band)
        ma = moving_average(time_averaged, 20)
        ma_max = moving_average(time_max, 20)
        ma_min = moving_average(time_min, 20)
        ma_modal = moving_average(modal_values, 20)
        ma_std = moving_average(one_std, 20)
        ma_seventyth = moving_average(seventyth_percent, 20)
        ma_global.append(ma)
        ma_modal_global.append(ma_modal)
        ma_std_global.append(ma_std)
        ma_seventyth_global.append(ma_seventyth)

    region_averaged_ma = defaultdict(list)
    region_averaged_modal = defaultdict(list)
    region_averaged_std = defaultdict(list)
    region_averaged_seventyth = defaultdict(list)

    for j in range(64):
        region = ''.join([k for k in ch_names[j] if not k.isdigit()])
        region_averaged_ma[region].append(ma_global[j])
        region_averaged_modal[region].append(ma_modal_global[j])
        region_averaged_std[region].append(ma_std_global[j])
        region_averaged_seventyth[region].append(ma_seventyth_global[j])
    keys = region_averaged_ma.keys()
    [row, column] = get_subplot_dimensions(keys)
    fig_size = 1 * len(keys)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    for key in region_averaged_ma:
        ma_avg = np.mean(region_averaged_ma[key], axis=0)
        mode_avg = np.mean(region_averaged_modal[key], axis=0)
        std_avg = np.mean(region_averaged_std[key], axis=0)
        seventyth_avg = np.mean(region_averaged_seventyth[key], axis=0)
        ax[active_row, active_column].plot(ma_avg, label='MA Mean')
        ax[active_row, active_column].plot(mode_avg, label='MA Binned Mode')
        ax[active_row, active_column].plot(std_avg, label='MA one std')
        ax[active_row, active_column].plot(seventyth_avg, label='MA 70th')
        ax[active_row, active_column].legend()
        ax[active_row, active_column].set_title(key)
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0

    if not save:
        plt.show()
    else:
        # plt.ioff()
        plt.savefig(filename)
        plt.close(fig)


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
        # data_to_plot = data_to_plot[::2]
        # downsampling = SAMPLING_SPEED / 2
        w = 6.
        # freq = np.linspace(1, 50, 200)
        # freq = np.geomspace(5, 14, 5)
        freq = np.linspace(23, 24, 2)
        widths = w * SAMPLING_SPEED / (2 * freq * np.pi)
        cwtm = signal.cwt(data_to_plot, signal.morlet2, widths, w=w)
        # print(len(np.abs(cwtm[:, ::100])))
        # print(cwtm[:, ::100])
        magnitudes = np.abs(cwtm[:, ::100])  # Maybe average instead ...
        t = np.linspace(1, len(data_to_plot), len(magnitudes[0]))
        averaged_band = np.mean(magnitudes, axis=0)
        time_averaged, time_max, time_min, one_std, modal_values, seventyth_percent = extract_averages(averaged_band)
        ax[active_row, active_column].plot(time_averaged, 'o', label='Raw Mean')
        ma = moving_average(time_averaged, 20)
        ma_max = moving_average(time_max, 20)
        ma_min = moving_average(time_min, 20)
        ma_modal = moving_average(modal_values, 20)
        ma_std = moving_average(one_std, 20)
        ma_seventyth = moving_average(seventyth_percent, 20)
        ax[active_row, active_column].plot(ma, label='MA Mean')
        ax[active_row, active_column].plot(ma_modal, label='MA Binned Mode')
        ax[active_row, active_column].plot(ma_std, label='MA one std')
        ax[active_row, active_column].plot(ma_seventyth, label='MA 70th')
        ax[active_row, active_column].legend()
        # ax[active_row, active_column].plot(ma_min)

        max_lim = 2 * np.std(time_averaged) + np.mean(time_averaged)
        min_lim = np.mean(time_averaged) - np.std(time_averaged)

        # pc = ax[active_row, active_column].pcolormesh(t, freq, magnitudes, cmap='viridis', shading='gouraud')
        # plt.colorbar(pc, ax=ax[active_row, active_column])
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


def get_fft(data, sampling_rate):
    fft_data = fft(data)
    N = len(fft_data)
    n = np.arange(N)
    T = N / sampling_rate
    freq = n / T
    return [fft_data, freq]


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


def extract_averages(raw_data_values, window_averaging=20):
    time_averaged = []
    time_max = []
    time_min = []
    one_std = []
    modal_values = []
    seventyth_percent = []
    # print(len(raw_data_values))
    for j in range(0, len(raw_data_values), window_averaging):
        if j + window_averaging < len(raw_data_values):
            end_index = j + window_averaging
        else:
            end_index = len(raw_data_values) - 1
        values = raw_data_values[j:end_index]
        time_averaged.append(np.mean(values))
        one_std.append(np.mean(values) + np.std(values))
        seventyth_percent.append(np.percentile(values, 70))
        time_max.append(np.max(values))
        time_min.append(np.min(values))
        bins = np.linspace(0, np.max(values), 10)
        indices = np.digitize(values, bins)
        modal_index = mode(indices)
        # print(modal_index)
        modal_midpoint = (bins[modal_index[0]] + bins[modal_index[0] - 1]) / 2
        modal_values.append(modal_midpoint[0])
    return time_averaged, time_max, time_min, one_std, modal_values, seventyth_percent


def stft_test(eeg_data, electrodes_to_plot, np_slice_indexes, save=False, filename=None, plot_averaged=False,
              band='beta'):
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
                band_values = []
                for z in range(len(f)):
                    if 28 >= f[z] >= 15 and band == 'beta':
                        band_values.append(abs_power[z, j])
                    if 13 >= f[z] >= 8 and band == 'alpha':
                        band_values.append(abs_power[z, j])
                    if 8 >= f[z] >= 4 and band == 'theta':
                        band_values.append(abs_power[z, j])
                    if 25 >= f[z] >= 23 and band == 'beta_entrain':
                        band_values.append(abs_power[z, j])
                    if 19 >= f[z] >= 17 and band == 'beta_entrain_low':
                        band_values.append(abs_power[z, j])
                ave_alpha = np.mean(band_values)  # Decide on what to use here
                alpha_average_values.append(ave_alpha)
            time_averaged = []
            time_max = []
            time_min = []
            one_std = []
            modal_values = []
            seventyth_percent = []
            window_averaging = 20
            for j in range(0, len(alpha_average_values), window_averaging):
                if j + window_averaging < len(alpha_average_values):
                    end_index = j + window_averaging
                else:
                    end_index = len(alpha_average_values) - 1
                if j == end_index:
                    j = len(alpha_average_values)
                    continue
                values = alpha_average_values[j:end_index]
                time_averaged.append(np.mean(values))
                one_std.append(np.mean(values) + np.std(values))
                seventyth_percent.append(np.percentile(values, 70))
                time_max.append(np.max(values))
                time_min.append(np.min(values))
                bins = np.linspace(0, np.max(values), 10)
                indices = np.digitize(values, bins)
                modal_index = mode(indices)
                # print(modal_index)
                modal_midpoint = (bins[modal_index[0] - 1] + bins[modal_index[0] - 2]) / 2
                modal_values.append(modal_midpoint[0])
            ax[active_row, active_column].plot(time_averaged, 'o', label='Raw Mean')
            ma = moving_average(time_averaged, 20)
            ma_max = moving_average(time_max, 20)
            ma_min = moving_average(time_min, 20)
            ma_modal = moving_average(modal_values, 20)
            ma_std = moving_average(one_std, 20)
            ma_seventyth = moving_average(seventyth_percent, 20)
            ax[active_row, active_column].plot(ma, label='MA Mean')
            ax[active_row, active_column].plot(ma_modal, label='MA Binned Mode')
            ax[active_row, active_column].plot(ma_std, label='MA one std')
            ax[active_row, active_column].plot(ma_seventyth, label='MA 70th')
            ax[active_row, active_column].legend()
            # ax[active_row, active_column].plot(ma_min)

            max_lim = 2 * np.std(time_averaged) + np.mean(time_averaged)
            min_lim = np.mean(time_averaged) - np.std(time_averaged)
            # ax[active_row, active_column].set(ylim=[min_lim, max_lim])
        else:
            ax[active_row, active_column].pcolormesh(t, f[5:31], np.abs(Zxx[5:31]), cmap='viridis', shading='gouraud')
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


# TODO split plot colour by entrainment frequency
def stft_by_region(eeg_data, electrodes_to_plot, np_slice_indexes, band='beta', artifact_epochs=None, save_plot=False,
                   filename=None, save_values=False):
    active_row = 0
    active_column = 0
    ma_global = []
    ma_time_global = []
    ml_epochs = []
    data = eeg_data.get_data().transpose()
    for i in electrodes_to_plot:
        data_to_plot = data[np_slice_indexes[i]]
        ml_epochs = [x * SAMPLING_SPEED * 3 * 60 for x in range(1, 5)]
        if artifact_epochs:
            for epoch in artifact_epochs:
                index_low = math.floor(epoch[0] * SAMPLING_SPEED)
                index_high = math.ceil(epoch[1] * SAMPLING_SPEED)
                data_to_plot = np.delete(data_to_plot, slice(index_low, index_high))
                samples_removed = index_high - index_low
                affecting_ml_epoch = False
                for idx_ml_epoch, ml_epoch in enumerate(ml_epochs):
                    if affecting_ml_epoch:
                        ml_epochs[idx_ml_epoch] -= samples_removed
                        continue
                    if index_low < ml_epoch and index_high < ml_epoch:
                        ml_epochs[idx_ml_epoch] -= samples_removed
                        affecting_ml_epoch = True
                    elif index_low < ml_epoch and index_high > ml_epoch:
                        ml_epochs[idx_ml_epoch] -= (samples_removed + index_high - ml_epoch)
                        affecting_ml_epoch = True

        samples_per_ft = 100
        overlap = 10
        f, t, Zxx = signal.stft(x=data_to_plot, fs=SAMPLING_SPEED, nperseg=samples_per_ft, noverlap=overlap, nfft=512)
        abs_power = np.abs(Zxx)
        band_averaged_values = []
        for j in range(len(abs_power[0])):
            band_values = []
            for z in range(len(f)):
                if 28 >= f[z] >= 15 and band == 'beta':
                    band_values.append(abs_power[z, j])
                if 13 >= f[z] >= 8 and band == 'alpha':
                    band_values.append(abs_power[z, j])
                if 8 >= f[z] >= 4 and band == 'theta':
                    band_values.append(abs_power[z, j])
                if 25 >= f[z] >= 23 and band == 'beta_entrain':
                    band_values.append(abs_power[z, j])
                if 19 >= f[z] >= 17 and band == 'beta_entrain_low':
                    band_values.append(abs_power[z, j])
            ave_alpha = np.mean(band_values)  # Decide on what to use here
            band_averaged_values.append(ave_alpha)
        time_averaged = []
        time_avg = []
        window_averaging = 20
        for j in range(0, len(band_averaged_values), window_averaging):
            if j + window_averaging < len(band_averaged_values):
                end_index = j + window_averaging
            else:
                end_index = len(band_averaged_values) - 1
            if j == end_index:
                break
            values = band_averaged_values[j:end_index]
            time_values = t[j:end_index]
            time_avg.append(np.mean(time_values))
            time_averaged.append(np.mean(values))
        ma = moving_average(time_averaged, 20)
        ma_time = moving_average(time_avg, 20)
        ma_time = [x / 60 for x in ma_time]
        ma_time_global = ma_time
        ma_global.append(ma)

    region_averaged_ma = defaultdict(list)

    for j in range(62):
        region = ''.join([k for k in ch_names[j] if not k.isdigit()])
        region_averaged_ma[region].append(ma_global[j])
    if save_plot:
        keys = region_averaged_ma.keys()
        [row, column] = get_subplot_dimensions(keys)
        fig_size = 1 * len(keys)
        fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
        fig.tight_layout(pad=0.5)  # edit me when axis labels are added
        fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    for key in region_averaged_ma:
        ma_avg = np.mean(region_averaged_ma[key], axis=0)
        x_end = ma_time_global[-1]
        y_end = ma_avg[-1]
        y_max_index = np.argmax(ma_avg)
        y_min_index = np.argmin(ma_avg)
        label_end = "{:.4f}".format(y_end)
        label_start = "{:.4f}".format(ma_avg[0])
        if save_plot:
            ax[active_row, active_column].plot(ma_time_global, ma_avg, label='MA Mean')
            if y_max_index != 0 and y_max_index != len(ma_avg) - 1:
                label_max = "{:.4f}".format(ma_avg[y_max_index])
                ax[active_row, active_column].annotate(label_max, (ma_time_global[y_max_index], ma_avg[y_max_index]),
                                                       textcoords="offset points",
                                                       xytext=(0, 5),
                                                       ha='left')
                ax[active_row, active_column].plot(ma_time_global[y_max_index], ma_avg[y_max_index], 'r.')
            if y_min_index != 0 and y_min_index != len(ma_avg) - 1:
                label_min = "{:.4f}".format(ma_avg[y_min_index])
                ax[active_row, active_column].annotate(label_min, (ma_time_global[y_min_index], ma_avg[y_min_index]),
                                                       textcoords="offset points",
                                                       xytext=(0, 10),
                                                       ha='left')
                ax[active_row, active_column].plot(ma_time_global[y_min_index], ma_avg[y_min_index], 'r.')
            ax[active_row, active_column].annotate(label_end, (x_end, y_end), textcoords="offset points", xytext=(0, 10),
                                                   ha='right')
            ax[active_row, active_column].plot(x_end, y_end, 'r.')
            ax[active_row, active_column].annotate(label_start, (ma_time_global[0], ma_avg[0]), textcoords="offset points", xytext=(0, 10),
                                                   ha='left')
            ax[active_row, active_column].plot(ma_time_global[0], ma_avg[0], 'r.')
        # csv: participantID, dataset name, band filtered to, region, start value, end value, max,
        # min, 3 min, 6 min, 9 min, 12 min,
        csv_write_region = f'{filename.split("_")[0]}, {filename.split("_")[1]}, {band}'
        if key == 'T' or key == 'TP' or key == 'FT':
            csv_write_region += f', {key}'
            csv_write_region += f', {ma_avg[0]}'
            csv_write_region += f', {ma_avg[-1]}'
            csv_write_region += f', {ma_avg[y_max_index]}'
            csv_write_region += f', {ma_avg[y_min_index]}'

            if save_values:
                # np save
                array_to_save = [ma_time_global, ma_avg]
                np_ds_filename_data = f'stft_region_averaged/ft_values/{filename}'
                np.save(np_ds_filename_data, array_to_save)
                np_ds_filename_epochs = f'stft_region_averaged/epochs/{filename}'
                np.save(np_ds_filename_epochs, ml_epochs)

            for epoch in ml_epochs:
                # Fix
                epoch_in_s = epoch / (512 * 60)
                index_boolean = [t >= epoch_in_s for t in ma_time_global]
                index = np.argmax(index_boolean)
                epoch_value = ma_avg[index]
                csv_write_region += f', {epoch_value}'
                label_epoch = "{:.4f}".format(epoch_value)
                if save_plot:
                    ax[active_row, active_column].annotate(label_epoch, (ma_time_global[index], epoch_value),
                                                           textcoords="offset points", xytext=(-15, -10),
                                                           ha='left')
                    ax[active_row, active_column].plot(ma_time_global[index], epoch_value, 'r.')
            csv_write_region += "\n"
            text_file = open("power_summary.csv", "a")
            text_file.write(csv_write_region)
            text_file.close()
        if save_plot:
            ax[active_row, active_column].grid()
            ax[active_row, active_column].legend()
            ax[active_row, active_column].set_title(key)
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
    if save_plot:
        plt.savefig(filename)
        plt.close(fig)
        del fig


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
        ax[active_row, active_column].pcolormesh(t, f[11:], np.abs(Zxx[11:]), cmap='viridis',
                                                 shading='gouraud')  # vmin=0, vmax=amp,
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
