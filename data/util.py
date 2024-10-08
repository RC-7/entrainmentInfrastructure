import numpy as np
import h5py
from math import sqrt, ceil
import matplotlib.pyplot as plt
from numpy import genfromtxt


def get_subplot_dimensions(electrodes_to_plot):
    row = 0
    column = 0
    if np.abs((len(electrodes_to_plot) / 2) - 2) > 2:
        sqrt_plots = sqrt(len(electrodes_to_plot))
        row = ceil(sqrt_plots)
        column = ceil(sqrt_plots)
        if row * column - column >= len(electrodes_to_plot):
            row -= 1
    else:
        row = 2
        column = ceil(len(electrodes_to_plot) / 2)
    return [row, column]


def setup_figure(list_of_figures, fig_size_modifier=1):
    [row, column] = get_subplot_dimensions(list_of_figures)
    active_row = 0
    active_column = 0
    fig_size = fig_size_modifier * len(list_of_figures)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    return row, column, fig, ax


def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    ma = np.convolve(data, window, 'same')
    inaccurate_window = int(np.ceil(window_size / 2))
    ma = ma[inaccurate_window: -inaccurate_window]
    return ma


def figure_handling(fig, filename='', save_fig=False, ):
    if not save_fig:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close(fig)
        del fig


def get_hemisphere(ch_names):
    ch_hemisphere = {}
    hemisphere = ['right', 'left']
    ch: str
    for ch in ch_names:
        if not ch.isalpha():
            hemisphere_index = int(ch[-1]) % 2
            ch_hemisphere[ch] = hemisphere[hemisphere_index]
        else:
            ch_hemisphere[ch] = 'centre'
    return ch_hemisphere


def crop_data(raw_data, tmin_crop=None, tmax_crop=None):
    data = raw_data.copy()
    if tmax_crop is not None and tmin_crop is not None:
        data.crop(tmin=tmin_crop, tmax=tmax_crop).load_data()
    elif tmax_crop is None and tmin_crop is not None:
        data.crop(tmin=tmin_crop).load_data()
    return data


def save_hdfs5(filename, data):
    hf = h5py.File(filename, 'a')
    dataset_name = 'raw_data'
    hf.create_dataset(dataset_name, data=data, chunks=True, maxshape=(None, len(data[0])))
    hf.close()


def get_data_csv(filename):
    return genfromtxt(filename, delimiter=',')


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


def epoch_artifacts(raw_data, ch_names, threshold=None):
    sampling_speed = 512
    buffer = 2 * sampling_speed
    artefact_max_time = 10
    cz_index = ch_names.index('Cz')
    cz_data_index = np.index_exp[:, cz_index]
    eeg_data = raw_data.get_data().transpose()
    data = eeg_data[cz_data_index]
    indices_above_thresh = np.argwhere(data > threshold)
    epochs = []
    diff = np.diff(indices_above_thresh, axis=0)
    start = 0
    for idx_diff, diff_val in enumerate(diff):
        end_index = idx_diff == len(diff) - 1
        if start == 0:
            start = indices_above_thresh[idx_diff][0]
        elif diff_val != 1 or end_index:
            # Need to remove buffer created overlaps
            end_value = indices_above_thresh[idx_diff][0]
            start_epoch = (start - buffer) / sampling_speed
            end_epoch = (end_value + buffer) / sampling_speed
            if end_epoch - start_epoch > artefact_max_time:
                start = 0
                continue
            next_start_value = indices_above_thresh[idx_diff + 1][0]
            next_start_epoch = next_start_value / sampling_speed
            # print(idx_diff)
            if next_start_epoch > end_epoch or end_index:
                array_to_append = [start_epoch, end_epoch]
                epochs.append(array_to_append)
                start = 0
    # Group together epochs less than 1 s from one another
    final_epochs = []
    spaced_epoch = []
    grouping_threshold = 1
    for id_epoch, epoch in enumerate(epochs):
        if not spaced_epoch:
            spaced_epoch.append(epoch[0])
        if id_epoch != len(epochs) - 1:
            if np.abs(epochs[id_epoch + 1][0] - epoch[1]) > grouping_threshold:
                spaced_epoch.append(epoch[1])
        else:
            spaced_epoch.append(epoch[1])
        if len(spaced_epoch) == 2:
            final_epochs.append(spaced_epoch)
            spaced_epoch = []

    return final_epochs
