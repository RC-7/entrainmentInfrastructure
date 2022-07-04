from collections import defaultdict

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from constants import ch_names, eeg_bands, SAMPLING_SPEED, ch_hemisphere
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, get_window
from scipy.signal import hilbert

import networkx as nx

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
                correl = correl[0, 1] ** 2
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
                correl = correl[0] ** 2
            global_correl[set[2]].append(correl)
            if current_window_low <= time_sound * SAMPLING_SPEED:
                is_entrain[set[2]].append(0)
            else:
                is_entrain[set[2]].append(1)

    active_row = 0
    active_column = 0

    row, column, fig, ax = setup_figure(set_positions, fig_size_modifier=2)
    ma_period = 30
    for key in global_correl:
        correl_ma = moving_average(global_correl[key], ma_period)
        if time_sound > 0:
            if not correl_ma.any():
                continue
            entrain_clipped = np.clip(is_entrain[key], np.min(correl_ma), np.max(correl_ma))
            is_entrain_ma = moving_average(entrain_clipped, ma_period)
            ax[active_row, active_column].plot(correl_ma)
            ax[active_row, active_column].plot(is_entrain_ma)
        ax[active_row, active_column].set_title(key)
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    if active_column != 0:
        for j in range(active_column, column):
            fig.delaxes(ax[active_row, j])
    if active_row < row -1:
        for j in range(0, column):
            fig.delaxes(ax[row - 1, j])
    figure_handling(fig, filename, save_fig)


def phase_locking_value(raw_data, electrodes_to_plot, method='hilbert', save_ds=False, filename=None, ratio=False):
    raw_data = raw_data.get_data().transpose()
    max_sample = len(raw_data[:, 0])
    window = 1024
    instantaneous_phase_global = defaultdict(list)
    for i in range(len(electrodes_to_plot)):
        for current_window_low in range(0, max_sample, window):
            index_ch = np.index_exp[current_window_low:current_window_low + window, i]
            # Need to add logic here... maybe padd with zeros
            # if (np.max(raw_data[index_ch]) - np.min(raw_data[index_ch])) > 110 or \
            #         (np.max(raw_data[index_ch]) - np.min(raw_data[index_ch])) > 110:
            #     continue
            if method == 'hilbert':
                # Try for power too
                analytic_signal = hilbert(raw_data[index_ch])
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_phase_global[i] = np.append(instantaneous_phase_global[i], instantaneous_phase.copy())
    plv_global = defaultdict(list)
    for i in instantaneous_phase_global.keys():
        for j in instantaneous_phase_global.keys():
            if i == j:
                continue
            phase_difference = np.subtract(instantaneous_phase_global[i], instantaneous_phase_global[j])
            f = lambda x: x % 2 * np.pi
            mod_phase = f(phase_difference)
            complex_phase_diff = np.exp(np.complex(0, 1) * mod_phase)
            plv = []
            trial_length = 512 * 5  # Look at me
            # Make if ratio
            k_start = 0
            if ratio:
                ref_plv = np.abs(np.sum(complex_phase_diff[0:SAMPLING_SPEED]) / SAMPLING_SPEED)
                plv.append(ref_plv)
                k_start = 512
            for k in range(k_start, len(complex_phase_diff), trial_length):
                # plv.append(np.abs(np.sum(complex_phase_diff[k:k + trial_length]) / (ref_plv * trial_length)))
                plv.append(np.abs(np.sum(complex_phase_diff[k:k + trial_length]) / trial_length))
            key = f'{ch_names[i]}-{ch_names[j]}'
            plv_global[key] = plv
    if save_ds:
        ds_filename = f'PLV_dataset/{filename}'
        np.save(ds_filename, plv_global)
    return plv_global


def small_world(raw_data, electrodes_to_plot, method='hilbert', save_fig=False, filename='',
                plv=None):
    if plv is None:
        plv = phase_locking_value(raw_data, electrodes_to_plot, method=method, save_ds=True, filename=filename)
    sigma_values = []
    omega_values = []
    electrode_graph = nx.Graph()
    number_dp = len(plv[f'{ch_names[0]}-{ch_names[1]}'])
    for ch in range(len(electrodes_to_plot)):
        electrode_graph.add_node(ch_names[ch])
    for i in range(0, number_dp, 10):
        print(i)
        plv_values = [plv[j][i] for j in plv.keys()]
        medial_plv = np.median(plv_values)
        std_plv = np.std(plv_values)
        thresh = medial_plv + std_plv
        for connection in plv.keys():
            nodes = connection.split('-')
            electrode_graph.add_edge(nodes[0], nodes[1], weight=plv[connection][i])
        bad_conn_edges = list(filter(lambda e: e[2] < thresh,
                                     (e for e in electrode_graph.edges.data('weight'))))
        bad_conn_edges_ids = list(e[:2] for e in bad_conn_edges)
        electrode_graph.remove_edges_from(bad_conn_edges_ids)

        if electrode_graph.number_of_edges() > 1 and nx.is_connected(electrode_graph):
            print(electrode_graph.number_of_edges())
            sigma_values.append(nx.sigma(electrode_graph, niter=2, nrand=6))
        else:
            sigma_values.append(0)
        # omega_values.append(nx.omega(electrode_graph, niter=2, nrand=3))
        # print('omega done')
    sigma_ma = moving_average(sigma_values, 20)
    # omega_ma = moving_average(omega_values, 20)
    fig, ax = plt.subplots()
    ax.plot(sigma_values, marker='o')
    ax.set_title('Sigma')
    # ax[1].plot(omega_ma)
    # ax[1].set_title('Omega')
    figure_handling(fig, filename, save_fig)


def networkx_analysis(raw_data, electrodes_to_plot, method='hilbert', metric='clustering', save_fig=False, filename='',
                      plv=None, inter_hemisphere=False, ratio=False, entrain_time=0):
    if plv is None:
        plv = phase_locking_value(raw_data, electrodes_to_plot, method=method, save_ds=True, filename=filename,
                                  ratio=ratio)
    number_dp = len(plv[f'{ch_names[0]}-{ch_names[1]}'])
    metric_values_global = np.zeros((len(electrodes_to_plot), number_dp))
    electrode_graph = nx.Graph()
    for ch in range(len(electrodes_to_plot)):
        electrode_graph.add_node(ch_names[ch])
    for i in range(number_dp):
        for connection in plv.keys():
            nodes = connection.split('-')
            if inter_hemisphere and ch_hemisphere[nodes[0]] == ch_hemisphere[nodes[1]]:
                electrode_graph.add_edge(nodes[0], nodes[1], weight=0)
            else:
                electrode_graph.add_edge(nodes[0], nodes[1], weight=plv[connection][i])
        if metric == 'clustering':
            clustering_coefficients = list(nx.clustering(electrode_graph, weight='weight').values())
            metric_values_global[:, i] = clustering_coefficients
        elif metric == 'betweenness':
            betweenness_coeffcicients = list(nx.betweenness_centrality(electrode_graph, normalized=True,
                                                                       weight='weight').values())
            metric_values_global[:, i] = betweenness_coeffcicients
    row, column, fig, ax = setup_figure(electrodes_to_plot)
    active_row = 0
    active_column = 0
    for i in range(len(electrodes_to_plot)):
        cluster_single_electrode = metric_values_global[i, :]
        # Need to take into account trial averaging / summation in plv calc
        if ratio:
            mean_resting = cluster_single_electrode[0]
            get_ratio = lambda v: v / mean_resting
            cluster_single_electrode = np.array([get_ratio(x) for x in cluster_single_electrode])

        ma_cluster = moving_average(cluster_single_electrode, 20)
        ax[active_row, active_column].plot(ma_cluster)
        ax[active_row, active_column].set_title(ch_names[i])
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    figure_handling(fig, filename, save_fig)


def degree(raw_data, electrodes_to_plot, method='hilbert', save_fig=False, filename='', plv=None,
           inter_hemisphere=False):
    if plv is None:
        plv = phase_locking_value(raw_data, electrodes_to_plot, method=method)
    active_row = 0
    active_column = 0
    row, column, fig, ax = setup_figure(electrodes_to_plot)
    for i in range(len(electrodes_to_plot)):
        electrode_phase_locking = []
        for j in range(len(electrodes_to_plot)):
            if i == j:
                continue
            if inter_hemisphere and ch_hemisphere[ch_names[i]] == ch_hemisphere[ch_names[j]]:
                continue
            key = f'{ch_names[i]}-{ch_names[j]}'
            electrode_phase_locking.append(plv[key])
        degree = np.sum(electrode_phase_locking, axis=0)
        ma_degree = moving_average(degree, 20)
        ax[active_row, active_column].plot(ma_degree)
        ax[active_row, active_column].set_title(ch_names[i])
        active_column += 1
        if active_column == column:
            active_row += 1
            active_column = 0
    figure_handling(fig, filename, save_fig)
