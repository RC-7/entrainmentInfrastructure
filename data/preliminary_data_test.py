import numpy as np
from matplotlib import pyplot as plt

from power_analsis import stft_test, stft_by_region
from mne_wrapper import generate_mne_raw_with_info
from generic_analysis import create_x_values
import os
from util import crop_data, view_data, epoch_artifacts
from coherence_analysis import correl_coeff_to_ref, correl_coeff_set, phase_locking_value, degree, networkx_analysis, \
    small_world
from constants import ch_names

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SAMPLING_SPEED = 512


def main():
    # #####################################
    # ##########  Setup  ###########
    # #####################################
    ds_name = 'beta_audio'
    # ds_name = 'beta_audio_retest'
    # ds_name = 'alpha_audio'
    # ds_name = 'pink_audio'
    # participants = ['Full_run_V', 'Full_run_A', 'Full_run_B', 'Full_run_El', 'Full_run_H', 'Full_run_Jasp',
    #                 'Full_run_D', 'Full_run_S', 'Full_run_Zo', 'Full_run_P']
    participants = ['Full_run_D']
    test = ['Full_run_V', 'Full_run_A', 'Full_run_S', 'Full_run_Jasp', 'Full_run_D']
    threshold = 100

    for p in participants:
        min_crop = 40
        if p in test:
            ds_name = 'ml_beta_audio'
            min_crop = 0
        else:
            ds_name = 'pink_audio'
        filename = f'custom_suite/{p}/{ds_name}.h5'
        file_type = 'hdfs'
        [raw, info] = generate_mne_raw_with_info(file_type, filename, reference=True, scope='')
        electrodes_to_plot = [x for x in range(62)]
        # electrodes_to_plot = [0, 10, 20, 30, 40, 51]
        index_dict = {}
        for i in electrodes_to_plot:
            index_dict[i] = np.index_exp[:, i]
        cropped_data = crop_data(raw, min_crop)

        epochs = epoch_artifacts(cropped_data, ch_names, threshold)
        print(epochs)

        fn = f'{p.split("_")[-1]}_Whole_Beta_Clustering'
        print(fn)
        # output_filename = f'custom_suite/Full_run/{ds_name}_cleaned_V1.h5'
        # tmin_crop = 500
        # tmax_crop = 550

        # view_data(cropped_data)

        # #####################################
        # ##########  Connectivity  ###########
        # #####################################
        # correl_coeff_to_ref(cropped_data, electrodes_to_plot, ref='Cz')
        # correl_coeff_set(cropped_data, method='coeff', time_sound=60, filename='V_23-25-test', save_fig=True)
        # phase_locking_value(cropped_data, electrodes_to_plot)
        plv = np.load(f'PLV_dataset/{fn}.npy', allow_pickle=True).item()
        # degree(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='H_Pink_Beta_filt_degree',
        #        inter_hemisphere=False, plv=plv)
        # small_world(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='D_small_world_all',
        #             plv=plv)
        # networkx_analysis(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, plv=plv,
        #                   filename=fn, inter_hemisphere=False, metric='clustering',
        #                   ratio=False, entrain_time=80, region_averaged=True)

        # correl_coeff_set(cropped_data, method='coeff', save_fig=True, filename='el_Pink_beta_filt_cluster', time_sound=80)

        # ####################################
        # #############  Power  ##############
        # ####################################
        # data = cropped_data.get_data().transpose()
        # test_psd(data, electrodes_to_plot, index_dict)
        # # raw_data.pick([ch_names[n] for n in range(0, 3)])
        # stft_test(cropped_data, electrodes_to_plot, index_dict, save=True,
        #           filename='J_STFT_Beta_power.png',
        #           plot_averaged=True)

        fn = f'{p.split("_")[-1]}_Whole_Alpha_corrected'
        # stft_by_region(cropped_data, electrodes_to_plot, index_dict, save=True, filename=fn,
        #                plot_averaged=True)


if __name__ == '__main__':
    main()
