import numpy as np
from mne_wrapper import generate_mne_raw_with_info
import os
from util import crop_data, view_data
from coherence_analysis import correl_coeff_to_ref, correl_coeff_set, phase_locking_value, degree, networkx_analysis, \
    small_world

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SAMPLING_SPEED = 512


def main():
    ds_name = 'beta_pls'
    filename = f'custom_suite/Full_run_Jasp/{ds_name}.h5'
    output_filename = f'custom_suite/Full_run/{ds_name}_cleaned_V1.h5'
    file_type = 'hdfs'
    [raw, info] = generate_mne_raw_with_info(file_type, filename, reference=False, scope='beta_entrain')
    electrodes_to_plot = [x for x in range(62)]
    index_dict = {}
    for i in electrodes_to_plot:
        index_dict[i] = np.index_exp[:, i]
    tmin_crop = 500
    tmax_crop = 550
    cropped_data = crop_data(raw, 60)
    # view_data(cropped_data)

    # #####################################
    # ##########  Connectivity  ###########
    # #####################################
    # correl_coeff_to_ref(cropped_data, electrodes_to_plot, ref='Cz')
    # correl_coeff_set(cropped_data, method='coeff', time_sound=60, filename='V_23-25-test', save_fig=True)
    # phase_locking_value(cropped_data, electrodes_to_plot)
    plv = np.load('PLV_dataset/Zo_pink.npy', allow_pickle=True).item()
    # degree(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='Zo_pink_degree_entranf',
    #        inter_hemisphere=False, plv=None)
    # small_world(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='D_small_world_all',
    #             plv=plv)
    networkx_analysis(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, plv=None,
                      filename='Jasp_Beta_BetaEntrain_Filt_clustering', inter_hemisphere=False, metric='clustering')

    # ####################################
    # #############  Power  ##############
    # ####################################
    # data = cropped_data.get_data().transpose()
    # test_psd(data, electrodes_to_plot, index_dict)
    # # raw_data.pick([ch_names[n] for n in range(0, 3)])
    # stft_test(cropped_data, electrodes_to_plot, index_dict, save=True,
    #           filename='A_beta_Beta_20-26_no_ref.png',
    #           plot_averaged=True)

    # stft_by_region(cropped_data, electrodes_to_plot, index_dict, save=True, filename='ST_stft_by_region.png',
    #                plot_averaged=True)

if __name__ == '__main__':
    main()
