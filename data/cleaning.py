import math
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
import emd
from generic_analysis import plot_filtered
from util import crop_data, save_hdfs5
from constants import SAMPLING_SPEED, ch_names


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

