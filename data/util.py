import numpy as np
from math import sqrt, ceil
import matplotlib.pyplot as plt

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


def setup_figure(list_of_figures):
    [row, column] = get_subplot_dimensions(list_of_figures)
    active_row = 0
    active_column = 0
    fig_size = 1 * len(list_of_figures)
    fig, ax = plt.subplots(row, column, figsize=(fig_size, fig_size))
    fig.tight_layout(pad=0.5)  # edit me when axis labels are added
    fig.tight_layout(pad=1.5)  # edit me when axis labels are added
    return row, column, fig, ax


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def figure_handling(fig, filename='', save_fig=False, ):
    if not save_fig:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close(fig)
