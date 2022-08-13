import math
from collections import defaultdict

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from constants import ch_names, eeg_bands, SAMPLING_SPEED, ch_hemisphere, coherence_analysis_file, percentage_coherence_analysis_file
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


def phase_locking_value(raw_data, electrodes_to_plot, method='hilbert', save_ds=False, filename=None, ratio=False,
                        artifact_epochs=None):
    raw_data = raw_data.get_data().transpose()
    max_sample = len(raw_data[:, 0])
    window = 1024
    instantaneous_phase_global = defaultdict(list)
    ml_epochs = [x * SAMPLING_SPEED * 3 * 60 for x in range(1, 5)]
    if artifact_epochs:
        for epoch in artifact_epochs:
            index_low = math.floor(epoch[0] * SAMPLING_SPEED)
            index_high = math.ceil(epoch[1] * SAMPLING_SPEED)
            # data_to_plot = np.delete(data_to_plot, slice(index_low, index_high))
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
    for i in range(len(electrodes_to_plot)):
        data_index = np.index_exp[:, i]
        data = raw_data[data_index]
        for epoch in artifact_epochs:
            index_low = math.floor(epoch[0] * SAMPLING_SPEED)
            index_high = math.ceil(epoch[1] * SAMPLING_SPEED)
            data = np.delete(data, slice(index_low, index_high))
            max_sample = len(data[:])
        for current_window_low in range(0, max_sample, window):
            index_ch = np.index_exp[current_window_low:current_window_low + window]
            # Need to add logic here... maybe padd with zeros
            # if (np.max(raw_data[index_ch]) - np.min(raw_data[index_ch])) > 110 or \
            #         (np.max(raw_data[index_ch]) - np.min(raw_data[index_ch])) > 110:
            #     continue
            if method == 'hilbert':
                analytic_signal = hilbert(data[index_ch])
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
        ds_filename = f'PLV/dataset/{filename}'
        np.save(ds_filename, plv_global)
        epoch_filename = f'PLV/epoch/{filename}'
        np.save(epoch_filename, ml_epochs)
    return plv_global, ml_epochs


def small_world(raw_data, electrodes_to_plot, method='hilbert', save_fig=False, filename=''):
    try:
        plv = np.load(f'PLV/dataset/{filename}.npy', allow_pickle=True).item()
        ml_epochs = np.load(f'PLV/epoch/{filename}.npy', allow_pickle=True).item()
    except OSError:
        plv, ml_epochs = phase_locking_value(raw_data, electrodes_to_plot, method=method, save_ds=True,
                                             filename=filename)

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
                      inter_hemisphere=False, ratio=False, entrain_time=0, region_averaged=False,
                      artifact_epochs=None, band='beta'):
    try:
        plv = np.load(f'PLV/dataset/{filename}.npy', allow_pickle=True).item()
        ml_epochs = np.load(f'PLV/epoch/{filename}.npy', allow_pickle=True)
    except OSError:
        plv, ml_epochs = phase_locking_value(raw_data, electrodes_to_plot, method=method, save_ds=True,
                                             filename=filename, artifact_epochs=artifact_epochs)
    number_dp = len(plv[f'{ch_names[0]}-{ch_names[1]}'])
    time_values = range(number_dp)
    # 5 s trials
    time_values = [((x * 5) / 60) for x in time_values]
    ma_time_global = moving_average(time_values, 20)
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
    if not region_averaged:
        row, column, fig, ax = setup_figure(electrodes_to_plot)
    active_row = 0
    active_column = 0
    region_averaged_aggregate = defaultdict(list)
    for i in range(len(electrodes_to_plot)):
        cluster_single_electrode = metric_values_global[i, :]
        # Need to take into account trial averaging / summation in plv calc
        if ratio:
            mean_resting = cluster_single_electrode[0]
            get_ratio = lambda v: v / mean_resting
            cluster_single_electrode = np.array([get_ratio(x) for x in cluster_single_electrode])
        ma_cluster = moving_average(cluster_single_electrode, 20)

        if not region_averaged:
            ax[active_row, active_column].plot(ma_time_global, ma_cluster)
            x_end = ma_time_global[-1]
            y_end = ma_cluster[-1]
            y_max_index = np.argmax(ma_cluster)
            y_min_index = np.argmin(ma_cluster)
            label_end = "{:.2f}".format(y_end)
            label_start = "{:.2f}".format(ma_cluster[0])
            label_max = "{:.2f}".format(ma_cluster[y_max_index])
            label_min = "{:.2f}".format(ma_cluster[y_min_index])
            ax[active_row, active_column].annotate(label_end, (x_end, y_end), textcoords="offset points",
                                                   xytext=(0, 10),
                                                   ha='right')
            ax[active_row, active_column].annotate(label_start, (ma_time_global[0], ma_cluster[0]),
                                                   textcoords="offset points", xytext=(0, 10),
                                                   ha='left')
            ax[active_row, active_column].annotate(label_max, (ma_time_global[y_max_index], ma_cluster[y_max_index]),
                                                   textcoords="offset points",
                                                   xytext=(0, 10),
                                                   ha='left')
            ax[active_row, active_column].annotate(label_min, (ma_time_global[y_min_index], ma_cluster[y_min_index]),
                                                   textcoords="offset points",
                                                   xytext=(0, 10),
                                                   ha='left')
            ax[active_row, active_column].set_title(ch_names[i])
            active_column += 1
            if active_column == column:
                active_row += 1
                active_column = 0
        else:
            region = ''.join([k for k in ch_names[i] if not k.isdigit()])
            region_averaged_aggregate[region].append(ma_cluster)
    if region_averaged:
        row, column, fig, ax = setup_figure(region_averaged_aggregate.keys())
        for key in region_averaged_aggregate:
            ma_avg = np.mean(region_averaged_aggregate[key], axis=0)
            ax[active_row, active_column].plot(ma_time_global, ma_avg)
            x_end = ma_time_global[-1]
            y_end = ma_avg[-1]
            y_max_index = np.argmax(ma_avg)
            y_min_index = np.argmin(ma_avg)
            label_end = "{:.4f}".format(y_end)
            label_start = "{:.4f}".format(ma_avg[0])
            if y_max_index != 0 and y_max_index != len(ma_avg) - 1:
                label_max = "{:.4f}".format(ma_avg[y_max_index])
                ax[active_row, active_column].annotate(label_max, (ma_time_global[y_max_index], ma_avg[y_max_index]),
                                                       textcoords="offset points",
                                                       xytext=(0, 2),
                                                       ha='right')
            if y_min_index != 0 and y_min_index != len(ma_avg) - 1:
                label_min = "{:.4f}".format(ma_avg[y_min_index])
                ax[active_row, active_column].annotate(label_min, (ma_time_global[y_min_index], ma_avg[y_min_index]),
                                                       textcoords="offset points",
                                                       xytext=(0, -5),
                                                       ha='right')
            ax[active_row, active_column].annotate(label_end, (x_end, y_end), textcoords="offset points",
                                                   xytext=(0, 10),
                                                   ha='right')
            ax[active_row, active_column].annotate(label_start, (ma_time_global[0], ma_avg[0]),
                                                   textcoords="offset points", xytext=(0, 10),
                                                   ha='left')

            csv_write_region = f'{filename.split("_")[0]}, {filename.split("_")[1]}, {band}'
            if key == 'T' or key == 'TP' or key == 'FT':
                csv_write_region += f', {key}'
                csv_write_region += f', {ma_avg[0]}'
                csv_write_region += f', {ma_avg[-1]}'
                csv_write_region += f', {ma_avg[y_max_index]}'
                csv_write_region += f', {ma_avg[y_min_index]}'

                count = 0
                for val in ma_avg:
                    if val > ma_avg[0]:
                        count += 1
                data_to_save = f'{filename.split("_")[0]}, {filename.split("_")[1]}, {band}, {key}, '
                data_to_save += str(count / len(ma_avg) * 100)
                data_to_save += '\n'

                text_file = open(percentage_coherence_analysis_file, "a")
                text_file.write(data_to_save)
                text_file.close()

                for epoch in ml_epochs:
                    # Fix
                    epoch_in_s = epoch / (512 * 60)
                    index_boolean = [t >= epoch_in_s for t in ma_time_global]
                    index = np.argmax(index_boolean)
                    epoch_value = ma_avg[index]
                    csv_write_region += f', {epoch_value}'
                    label_epoch = "{:.4f}".format(epoch_value)
                    if save_fig:
                        ax[active_row, active_column].annotate(label_epoch, (ma_time_global[index], epoch_value),
                                                               textcoords="offset points", xytext=(-15, -10),
                                                               ha='left')
                        ax[active_row, active_column].plot(ma_time_global[index], epoch_value, 'r.')
                csv_write_region += "\n"
                text_file = open(coherence_analysis_file, "a")
                text_file.write(csv_write_region)
                text_file.close()
            ax[active_row, active_column].set_title(key)
            ax[active_row, active_column].grid()
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
