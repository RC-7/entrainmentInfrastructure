from collections import defaultdict

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from constants import ch_names, eeg_bands, SAMPLING_SPEED
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, get_window

from util import get_subplot_dimensions, setup_figure, moving_average, figure_handling


def correl_coeff_to_ref(raw_data, electrodes_to_plot, ref='Cz'):
    raw_data = raw_data.get_data().transpose()
    max_sample = len(raw_data[:, 0])
    window = 250
    global_correl = defaultdict(list)
    for current_window_low in range(0, max_sample, window):
        for i in range(len(electrodes_to_plot)):
            index_ch = np.index_exp[current_window_low:current_window_low + window, i]
            index_cz = np.index_exp[current_window_low:current_window_low + window, ch_names.index(ref)]
            mean_value_fp1 = np.mean(raw_data[index_ch])
            std_div_fp1 = np.std(raw_data[index_ch])
            f = lambda x: np.abs(x - mean_value_fp1)
            displacement = f(raw_data[index_ch])
            if np.max(displacement) > mean_value_fp1 + 2.5 * std_div_fp1:
                break
            correl = np.corrcoef(raw_data[index_ch], raw_data[index_cz])
            global_correl[ch_names[i]].append(correl[0, 1])
    row, column, fig, ax = setup_figure(electrodes_to_plot)
    active_row = 0
    active_column = 0
    for key in global_correl:
        correl_ma = moving_average(global_correl[key], 30)
        ax[active_row, active_column].plot(correl_ma)
        ax[active_row, active_column].set_title(key)
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0


def correl_coeff_set(raw_data, method='coeff', save_fig=False, filename='', time_sound=0):
    raw_data = raw_data.get_data().transpose()
    max_sample = len(raw_data[:, 0])
    if method == 'magSquared':
        window = 1024
    else:
        window = 512
    global_correl = defaultdict(list)
    set_positions = [
        ['Fp1', 'Fp2', 'Fp1-Fp2'],
        ['F3', 'F4', 'F3-F4'],
        ['F7', 'F8', 'F7-F8'],
        ['C3', 'C4', 'C3-C4'],
        ['T7', 'T8', 'T7-T8'],
        ['P3', 'P4', 'P3-P4'],
        ['O1', 'O2', 'O1-O2'],
        ['PO3', 'PO4', 'PO3-PO4'],
        ['AF7', 'AF8', 'AF7-AF8'],
        ['FT7', 'FT8', 'FT7-FT8'],
        ['FC1', 'FC2', 'FC1-FC2']
    ]
    # plot_count = 0
    is_entrain = defaultdict(list)
    for current_window_low in range(0, max_sample, window):

        for set in set_positions:
            index_ch = np.index_exp[current_window_low:current_window_low + window, ch_names.index(set[0])]
            index_cz = np.index_exp[current_window_low:current_window_low + window, ch_names.index(set[1])]
            # mean_value_fp1 = np.mean(raw_data[index_ch])
            # std_div_fp1 = np.std(raw_data[index_ch])
            # f = lambda x: np.abs(x - mean_value_fp1)
            # displacement = f(raw_data[index_ch])
            if (np.max(raw_data[index_ch]) - np.min(raw_data[index_ch])) > 110 or \
                    (np.max(raw_data[index_cz]) - np.min(raw_data[index_cz])) > 110:
                # if plot_count < 5:
                #     fig, ax = plt.subplots(1)
                #     ax.plot(raw_data[index_cz])
                # plot_count += 1
                continue
            if method == 'coeff':
                correl = np.corrcoef(raw_data[index_ch], raw_data[index_cz])
                correl = correl[0, 1]**2
            elif method == 'magSquared':
                length_fft = 150
                if length_fft > len(raw_data[index_cz]):
                    continue
                hann_window = get_window('hann', Nx=length_fft)
                f, c_xy = coherence(raw_data[index_ch], raw_data[index_cz], SAMPLING_SPEED, nperseg=length_fft,
                                    nfft=SAMPLING_SPEED, window=hann_window)
                alpha_values = []
                for j in range(len(c_xy)):
                    if 25 >= f[j] >= 22:
                        alpha_values.append(c_xy[j])
                correl = np.mean(alpha_values)  # Decide on what to use here
            else:
                correl = pearsonr(raw_data[index_ch], raw_data[index_cz])
                correl = correl[0]**2
            global_correl[set[2]].append(correl)
            if current_window_low <= time_sound*SAMPLING_SPEED:
                is_entrain[set[2]].append(0)
            else:
                is_entrain[set[2]].append(1)

    active_row = 0
    active_column = 0

    row, column, fig, ax = setup_figure(set_positions)
    ma_period = 30
    for key in global_correl:
        correl_ma = moving_average(global_correl[key], ma_period)
        if time_sound > 0:
            entrain_clipped = np.clip(is_entrain[key], np.min(correl_ma), np.max(correl_ma))
            is_entrain_ma = moving_average(entrain_clipped, ma_period)
            ax[active_row, active_column].plot(correl_ma)
            ax[active_row, active_column].plot(is_entrain_ma)
        ax[active_row, active_column].set_title(key)
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    figure_handling(fig, filename, save_fig)