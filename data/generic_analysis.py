
import matplotlib.pyplot as plt
import numpy as np

from util import get_subplot_dimensions
from constants import SAMPLING_SPEED
from filtering import butter_highpass_filter

def create_x_values(data, sampling_speed=SAMPLING_SPEED):
    diff = 1 / sampling_speed
    x_val = []
    for i in range(len(data)):
        x_val.append(i * diff)
    return x_val


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
