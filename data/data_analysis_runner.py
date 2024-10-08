import numpy as np
from matplotlib import pyplot as plt

from power_analysis import stft_test, stft_by_region, analyse_power_values, order_percentrage_power
from mne_wrapper import generate_mne_raw_with_info, plot_sensor_locations
from generic_analysis import create_x_values
import os
from util import crop_data, view_data, epoch_artifacts
from coherence_analysis import correl_coeff_to_ref, correl_coeff_set, phase_locking_value, degree, networkx_analysis, \
    small_world
from constants import ch_names, power_analysis_file, percentage_coherence_analysis_file, coherence_analysis_file, \
    percentage_power_analysis_file

from participant_info import participants, test, increase_max_crop, custom_min_crop, reduced_min_crop

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SAMPLING_SPEED = 512


def main():
    # #####################################
    # ##########  Setup  ##################
    # #####################################
    # TODO thread this
    text_file = open(power_analysis_file, "w")
    power_summary_columns = "participantID, dataset_name, band, region, start, end, max, min, three_min, six_min, " \
                            "nine_min, twelve_min\n"
    text_file.write(power_summary_columns)
    text_file.close()


    text_file = open(coherence_analysis_file, "w")
    power_summary_columns = "participantID, dataset_name, band, region, start, end, max, min, three_min, six_min, " \
                            "nine_min, twelve_min\n"
    text_file.write(power_summary_columns)
    text_file.close()

    text_file = open(percentage_power_analysis_file, "w")
    power_summary_columns = "Participant, dataset, band, group, region, 3, 6, 9, 12, 15, average\n"
    text_file.write(power_summary_columns)
    text_file.close()

    text_file = open(percentage_coherence_analysis_file, "w")
    power_summary_columns = "Participant, dataset, band, region, %above\n"
    text_file.write(power_summary_columns)
    text_file.close()

    threshold = 90
    group = ''
    run_epoch_artifacts = True
    for p in participants:
        # TODO incorporate crop into 3 min buckets
        min_crop = 0
        max_crop = None
        # TODO Scope to datasets
        if p in increase_max_crop:
            max_crop = 60 * 14
        if p in test:
            group = 'test'
            if p == custom_min_crop:
                ds_names = ['ml_beta_audio', 'beta_audio', 'pink_audio']
                min_crop = 60
            else:
                ds_names = ['ml_beta_audio', 'beta_audio', 'pink_audio']
                min_crop = 15
        else:
            group = 'control'
            ds_names = ['pink_audio']
        file_type = 'hdfs'
        # Decide what Bands to run analysis on
        # bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
        bands = ['beta_entrain', 'beta_entrain_low']

        for ds_name in ds_names:
            if ds_name == 'beta_audio':
                min_crop = 40
            if ds_name == 'beta_audio' and p == custom_min_crop:
                min_crop = 60
            if ds_name == 'ml_beta_audio':
                min_crop = 15
            if ds_name == 'pink_audio' and p not in test:
                min_crop = 40
            if ds_name == 'pink_audio' and p in test:
                if p == reduced_min_crop:
                    min_crop = 40
            else:
                min_crop = 15
            filename = f'custom_suite/{p}/{ds_name}.h5'
            epochs = None
            electrodes_to_plot = [x for x in range(62)]
            index_dict = {}
            for i in electrodes_to_plot:
                index_dict[i] = np.index_exp[:, i]
            if run_epoch_artifacts:
                [raw, info] = generate_mne_raw_with_info(file_type, filename, reference=True, scope='')
                # Uncomment if want to plot sensor locations for data loaded
                # plot_sensor_locations(raw)

                cropped_data = crop_data(raw, min_crop, max_crop)
                epochs = epoch_artifacts(cropped_data, ch_names, threshold)

            for band in bands:
                if p in participants:
                    save_plot = False
                else:
                    save_plot = True
                if run_epoch_artifacts:
                    [raw, info] = generate_mne_raw_with_info(file_type, filename, reference=True, scope=band)
                    cropped_data = crop_data(raw, min_crop, max_crop)
                # Uncomment if want to view data using MNE
                # view_data(cropped_data)
                # #####################################
                # ##########  Connectivity  ###########
                # #####################################
                fn = f'{p.split("_")[-1]}_{ds_name}_{band}_Clustering'
                #
                networkx_analysis(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True,
                                  filename=fn, inter_hemisphere=False, metric='clustering',
                                  ratio=False, entrain_time=80, region_averaged=True, artifact_epochs=epochs, band=band)

                # ####################################
                # #############  Power  ##############
                # ####################################
                fn = f'{p.split("_")[-1]}_{ds_name}_{band}_filtered_Power'
                # TODO account for cropping time
                stft_by_region(cropped_data, electrodes_to_plot, index_dict, save_plot=save_plot, filename=fn,
                               artifact_epochs=epochs, band=band, save_values=True)
                analyse_power_values(fn, band, group, ds_name)
                fn = f'{p.split("_")[-1]}_{ds_name}_{band}_filtered_Power_no_avg'
                stft_test(cropped_data, electrodes_to_plot, index_dict, save=True, filename=fn, plot_averaged=True,
                          band=band)
    order_percentrage_power()


if __name__ == '__main__':
    main()

# Additional Power adn Connectivity analysis possible

# ####################################
# #############  Power  ##############
# ####################################
# data = cropped_data.get_data().transpose()
# test_psd(data, electrodes_to_plot, index_dict)
# # raw_data.pick([ch_names[n] for n in range(0, 3)])
# stft_test(cropped_data, electrodes_to_plot, index_dict, save=True,
#           filename='filename.png',
#           plot_averaged=True)


# #####################################
# ##########  Connectivity  ###########
# #####################################
# correl_coeff_to_ref(cropped_data, electrodes_to_plot, ref='Cz')
# correl_coeff_set(cropped_data, method='coeff', time_sound=60, filename='V_23-25-test', save_fig=True)
# phase_locking_value(cropped_data, electrodes_to_plot)

# degree(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='filename_Beta_filt_degree',
#        inter_hemisphere=False, plv=plv)
# small_world(cropped_data, electrodes_to_plot, method='hilbert', save_fig=True, filename='filename_small_world_all',
#             plv=plv)

# correl_coeff_set(cropped_data, method='coeff', save_fig=True, filename='filename_Pink_beta_filt_cluster', time_sound=80)
