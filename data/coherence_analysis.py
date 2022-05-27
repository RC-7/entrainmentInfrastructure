from collections import defaultdict

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from constants import ch_names, eeg_bands, SAMPLING_SPEED
import numpy as np
import matplotlib.pyplot as plt

# Do within frequncy range
from util import get_subplot_dimensions, setup_figure, moving_average, figure_handling


def correl_coeff_to_ref(raw_data, electrodes_to_plot, np_slice_indexes, fs=SAMPLING_SPEED, ref='Cz'):
    raw_data = raw_data.get_data().transpose()
    max_sample = len(raw_data[:, 0])
    current_window_low = 0
    window = 250
    global_correl = defaultdict(list)
    # while current_window_low + window <= max_sample:
    for current_window_low in range(0, max_sample, window):
        correlation_coefficients = []
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
            # correlation_coefficients.append(correl)
            global_correl[ch_names[i]].append(correl[0, 1])
            # print(i)
            # print(len(correlation_coefficients))
        # global_correl = np.append(global_correl, correlation_coefficients, axis=1)
        # global_correl.append(correlation_coefficients)

    [row, column] = get_subplot_dimensions(electrodes_to_plot)
    active_row = 0
    active_column = 0
    fig_size = 1 * len(electrodes_to_plot)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    for key in global_correl:
        # for i in range(len(electrodes_to_plot)):
        #     print(len(global_correl[:, i]))

        ax[active_row, active_column].plot(global_correl[key])
        ax[active_row, active_column].set_title(key)
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0


def correl_coeff_set(raw_data, method='coeff', save_fig=False, filename=''):
    raw_data = raw_data.get_data().transpose()
    max_sample = len(raw_data[:, 0])
    current_window_low = 0
    window = 250
    global_correl = defaultdict(list)
    set_positions = [
        ['Fp1', 'Fp2', 'Fp1-Fp2'],
        ['F3', 'F4', 'F3-F4'],
        ['F7', 'F8', 'F7-F8'],
        ['C3', 'C4', 'C3-C4'],
        ['T7', 'T8', 'T7-T8'],
        ['P3', 'P4', 'P3-P4'],
        ['O1', 'O2', 'O1-O2']
    ]

    # while current_window_low + window <= max_sample:
    for current_window_low in range(0, max_sample, window):
        correlation_coefficients = []
        for set in set_positions:
            index_ch = np.index_exp[current_window_low:current_window_low + window, ch_names.index(set[0])]
            index_cz = np.index_exp[current_window_low:current_window_low + window, ch_names.index(set[1])]
            mean_value_fp1 = np.mean(raw_data[index_ch])
            std_div_fp1 = np.std(raw_data[index_ch])
            f = lambda x: np.abs(x - mean_value_fp1)
            displacement = f(raw_data[index_ch])
            if np.max(displacement) > mean_value_fp1 + 2.5 * std_div_fp1:
                break
            if method == 'coeff':
                correl = np.corrcoef(raw_data[index_ch], raw_data[index_cz])
                correl = correl[0, 1]
            else:
                correl = pearsonr(raw_data[index_ch], raw_data[index_cz])
                correl = correl[0]
            # correlation_coefficients.append(correl)
            global_correl[set[2]].append(correl)
            # print(i)
            # print(len(correlation_coefficients))
        # global_correl = np.append(global_correl, correlation_coefficients, axis=1)
        # global_correl.append(correlation_coefficients)

    active_row = 0
    active_column = 0

    row, column, fig, ax = setup_figure(set_positions)

    for key in global_correl:
        correl_ma = moving_average(global_correl[key], 30)
        ax[active_row, active_column].plot(correl_ma)
        ax[active_row, active_column].set_title(key)
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    figure_handling(fig, filename, save_fig)