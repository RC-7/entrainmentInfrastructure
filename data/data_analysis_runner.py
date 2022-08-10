import numpy as np
from matplotlib import pyplot as plt

from power_analysis import stft_test, stft_by_region
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
    #                 'Full_run_D', 'Full_run_S', 'Full_run_Zo', 'Full_run_P', 'Full_run_J']
    # TODO NEED to pull actions and state into a set of csv files so that can analyse together!
    # TODO Then can also do a meta analysis of that and other csv produced
    # text_file = open("power_summary.csv", "w")
    # power_summary_columns = "participantID, dataset name, band filtered to, region, start value," \
    #                         " end value, max, min, 3 min, 6 min, 9 min, 12 min\n"
    # text_file.write(power_summary_columns)
    # text_file.close()
    participants = ['Full_run_T']
    # participants = ['Full_run_El', 'Full_run_P', 'Full_run_H', 'Full_run_Zo', 'Full_run_S', 'Full_run_A',
    #                 'Full_run_Jasp', 'Full_run_B']
    # participants = ['Full_run_T', 'Full_run_V', 'Full_run_St', 'Full_run_J', 'Full_run_D', 'Full_run_El', 'Full_run_P',
    #                 'Full_run_H', 'Full_run_Zo', 'Full_run_S', 'Full_run_A', 'Full_run_Jasp', 'Full_run_B']
    test = ['Full_run_V', 'Full_run_A', 'Full_run_S', 'Full_run_Jasp', 'Full_run_D', 'Full_run_J', 'Full_run_T']
    threshold = 90
    for p in participants:
        # TODO incorporate crop into 3 min buckets
        min_crop = 0
        max_crop = None
        # TODO Scope to datasets
        if p == 'Full_run_St' or p == 'Full_run_D':
            max_crop = 60 * 13
        if p in test:
            # Datasets pending
            # if p == 'Full_run_T':
            #     ds_names = ['beta_audio', 'pink_audio']
            #     min_crop = 40
            if p == 'Full_run_V':
                # ds_names = ['ml_beta_audio', 'beta_audio', 'pink_audio']
                min_crop = 60
            else:
                # ds_names = ['ml_beta_audio', 'beta_audio', 'pink_audio']
                ds_names = ['ml_beta_audio']
                min_crop = 15
        else:
            ds_names = ['pink_audio']
        file_type = 'hdfs'
        # bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
        bands = ['beta']

        for ds_name in ds_names:
            if ds_name == 'beta_audio':
                min_crop = 40
            if ds_name == 'beta_audio' and p == 'Full_run_V':
                min_crop = 60
            if ds_name == 'ml_beta_audio':
                min_crop = 15
            if ds_name == 'pink_audio' and p not in test:
                min_crop = 40
            if ds_name == 'pink_audio' and p in test:
                min_crop = 15
            filename = f'custom_suite/{p}/{ds_name}.h5'
            [raw, info] = generate_mne_raw_with_info(file_type, filename, reference=True, scope='')
            electrodes_to_plot = [x for x in range(62)]
            index_dict = {}
            for i in electrodes_to_plot:
                index_dict[i] = np.index_exp[:, i]
            cropped_data = crop_data(raw, min_crop, max_crop)
            epochs = epoch_artifacts(cropped_data, ch_names, threshold)

            for band in bands:
                # if p == 'Full_run_T' and ds_name in ['ml_beta_audio', 'beta_audio'] and band in ['beta', 'alpha',
                #                                                                                  'beta_entrain']:
                #     save_plot = False
                if p in ['Full_run_V', 'Full_run_D', 'Full_run_El', 'Full_run_P', 'Full_run_H',
                         'Full_run_Zo', 'Full_run_S', 'Full_run_A', 'Full_run_Jasp', 'Full_run_B', 'Full_run_T']:
                    save_plot = False
                else:
                    save_plot = True

                [raw, info] = generate_mne_raw_with_info(file_type, filename, reference=True, scope=band)
                cropped_data = crop_data(raw, min_crop, max_crop)
                # view_data(cropped_data)
                # #####################################
                # ##########  Connectivity  ###########
                # #####################################
                # fn = f'{p.split("_")[-1]}_Whole_Beta_Clustering'
                # print(fn)
                # plv = np.load(f'PLV_dataset/{fn}.npy', allow_pickle=True).item()
                # networkx_analysis(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, plv=plv,
                #                   filename=fn, inter_hemisphere=False, metric='clustering',
                #                   ratio=False, entrain_time=80, region_averaged=True)

                # ####################################
                # #############  Power  ##############
                # ####################################
                fn = f'{p.split("_")[-1]}_{ds_name}_{band}_filtered_Power'
                # TODO account for cropping time
                stft_by_region(cropped_data, electrodes_to_plot, index_dict, save_plot=save_plot, filename=fn,
                               artifact_epochs=epochs, band=band, save_values=True)
                # fn = f'{p.split("_")[-1]}_{ds_name}_{band}_filtered_Power_no_avg'
                # stft_test(cropped_data, electrodes_to_plot, index_dict, save=True, filename=fn, plot_averaged=True,
                #           band=band)


if __name__ == '__main__':
    main()

# ####################################
# #############  Power  ##############
# ####################################
# data = cropped_data.get_data().transpose()
# test_psd(data, electrodes_to_plot, index_dict)
# # raw_data.pick([ch_names[n] for n in range(0, 3)])
# stft_test(cropped_data, electrodes_to_plot, index_dict, save=True,
#           filename='J_STFT_Beta_power.png',
#           plot_averaged=True)


# #####################################
# ##########  Connectivity  ###########
# #####################################
# correl_coeff_to_ref(cropped_data, electrodes_to_plot, ref='Cz')
# correl_coeff_set(cropped_data, method='coeff', time_sound=60, filename='V_23-25-test', save_fig=True)
# phase_locking_value(cropped_data, electrodes_to_plot)

# degree(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='H_Pink_Beta_filt_degree',
#        inter_hemisphere=False, plv=plv)
# small_world(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='D_small_world_all',
#             plv=plv)

# correl_coeff_set(cropped_data, method='coeff', save_fig=True, filename='el_Pink_beta_filt_cluster', time_sound=80)
