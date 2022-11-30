import pandas as pd
from constants import power_analysis_file, coherence_analysis_file, percentage_coherence_analysis_file, \
    percentage_power_analysis_file, test_participants

import numpy as np
from clustering_analysis import k_means_analysis
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, ttest_ind_from_stats
from participant_info import test_participant_order, participants_short, test_short, control_short, \
    participants_to_include, p_exploit


def record_action_order():
    action_times_columns = ['participant_ID', '0', '3', '6', '9', '12', '15']
    action_times = ['0', '3', '6', '9', '12', '15']
    actions_taken = pd.DataFrame(columns=action_times_columns)
    optimal_actions_taken = pd.DataFrame(columns=action_times_columns)
    model_file = '../data/models/bciAgent_5'
    model = pd.read_csv(model_file, index_col=0)
    for idx, p in enumerate(test_participant_order):
        actions_file = f'../data/models/bciAgent_{idx}_actions.csv'
        actions = pd.read_csv(actions_file)
        actions_dict = {'participant_ID': p, action_times[0]: '24'}
        optimal_actions_dict = {'participant_ID': p, action_times[0]: 0}
        previous_action = '24'
        for j in range(0, len(action_times) - 1):
            action_taken = actions.loc[j]['action']
            actions_dict[action_times[j + 1]] = action_taken
            index_state = f'{actions.loc[j]["state"]}_{previous_action}'
            optimal_action = str(model.loc[index_state].idxmax(axis=0))
            previous_action = action_taken
            if str(action_taken) == optimal_action:
                optimal_actions_dict[action_times[j + 1]] = 1
            else:
                optimal_actions_dict[action_times[j + 1]] = 0
        actions_taken = actions_taken.append(actions_dict, ignore_index=True)
        optimal_actions_taken = optimal_actions_taken.append(optimal_actions_dict, ignore_index=True)
    optimal_actions_taken['optimal_actions'] = optimal_actions_taken[action_times[0:-1]].sum(axis=1)

    actions_taken.to_csv('meta_analysis/actions_taken', index=False)
    optimal_actions_taken.to_csv('meta_analysis/optimal_actions_taken', index=False)


def analyse_results(modality='power'):
    if modality == 'power':
        filename = power_analysis_file
        bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
    else:
        filename = coherence_analysis_file
        bands = ['beta_entrain', 'beta_entrain_low']

    power_df = pd.read_csv(filename, skipinitialspace=True)
    # columns = ['participantID', 'dataset', 'band', 'region', 'start', 'end', 'max', 'min', 'three_min',
    #            'six_min', 'nine_min', 'twelve_min']
    participants = participants_short
    test = test_short
    control = control_short

    power_df['group'] = power_df['participantID'].apply(lambda x: 'test' if x in test else 'control')

    power_df['overall'] = power_df['end'] - power_df['start']

    power_df['start_three'] = power_df['three_min'] - power_df['start']
    power_df['start_six'] = power_df['six_min'] - power_df['start']
    power_df['start_nine'] = power_df['nine_min'] - power_df['start']
    power_df['start_twelve'] = power_df['twelve_min'] - power_df['start']

    power_df['three_six'] = power_df['six_min'] - power_df['three_min']
    power_df['six_nine'] = power_df['nine_min'] - power_df['six_min']
    power_df['nine_twelve'] = power_df['twelve_min'] - power_df['nine_min']
    power_df['twelve_fifteen'] = power_df['end'] - power_df['twelve_min']

    power_df['increased_overall'] = power_df['overall'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_start_three'] = power_df['start_three'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_start_six'] = power_df['start_six'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_start_nine'] = power_df['start_nine'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_start_twelve'] = power_df['start_twelve'].apply(lambda x: 1 if x > 0 else 0)

    power_df['increased_three_six'] = power_df['three_six'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_six_nine'] = power_df['six_nine'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_nine_twelve'] = power_df['nine_twelve'].apply(lambda x: 1 if x > 0 else 0)
    power_df['increased_twelve_fifteen'] = power_df['twelve_fifteen'].apply(lambda x: 1 if x > 0 else 0)
    increasing_electrodes = pd.DataFrame(columns=['participantID', 'dataset', 'band', 'group', 'increased_overall',
                                                  'increased_start_three', 'increased_start_six', 'increased_start_nine'
        , 'increased_start_twelve', 'increased_three_six',
                                                  'increased_six_nine', 'increased_nine_twelve',
                                                  'increased_twelve_fifteen'])

    increased_count = ['increased_overall', 'increased_start_three', 'increased_start_six', 'increased_start_nine',
                       'increased_start_twelve', 'increased_three_six', 'increased_six_nine', 'increased_nine_twelve',
                       'increased_twelve_fifteen']
    for participant in participants:
        group = ''
        if participant in test:
            datasets = ['ml', 'beta', 'pink']
            group = 'test'
        else:
            datasets = ['pink']
            group = 'control'
        for dataset in datasets:
            for band in bands:
                participant_data = power_df[(power_df.participantID == participant) &
                                            (power_df.band == band)]
                ds_df = participant_data[participant_data.dataset_name == dataset]
                values = {
                    'participantID': participant,
                    'group': group,
                    'dataset': dataset,
                    'band': band,
                }
                for param in increased_count:
                    values[param] = ds_df[param].sum()

                increasing_electrodes = increasing_electrodes.append(values, ignore_index=True)
    increasing_electrodes.sort_values(by='dataset')

    datasets = ['ml', 'beta', 'pink']
    meta_increasing_columns = ['group', 'dataset', 'band', 'mean', 'metric', '0_count', '1_count', '2_count', '3_count',
                               'total increasing']
    meta_increasing = pd.DataFrame(columns=meta_increasing_columns)
    for band in bands:
        for dataset in datasets:
            if dataset == 'pink':
                groups = ['test', 'control', 'all']
            else:
                groups = ['test']

            for group in groups:
                if group == 'all':
                    ds_meta_increase = increasing_electrodes[(increasing_electrodes.dataset == dataset) &
                                                             (increasing_electrodes.band == band)]
                else:
                    ds_meta_increase = increasing_electrodes[(increasing_electrodes.group == group) &
                                                             (increasing_electrodes.dataset == dataset) &
                                                             (increasing_electrodes.band == band)]
                for metric in increased_count:
                    results_dict = {'group': group, 'dataset': dataset, 'metric': metric, 'band': band,
                                    'mean': ds_meta_increase[metric].mean()}
                    total_increasing = 0
                    for i in range(4):
                        increasing = len(ds_meta_increase[ds_meta_increase[metric] == i])
                        results_dict[f'{i}_count'] = increasing
                        if i > 0:
                            total_increasing += increasing
                    results_dict['total increasing'] = total_increasing
                    meta_increasing = meta_increasing.append(results_dict, ignore_index=True)

    statistics_across = pd.DataFrame(columns=['group', 'dataset', 'band', 'mean_start', 'mean_end', 'std_start',
                                              'std_end', 'mean_abs_diff', 'std_abs_diff', 'region'])

    regions = ['T', 'TP', 'FT', 'all']

    for region in regions:
        if region != 'all':
            scoped_electrode = power_df[(power_df.region == region)]
        else:
            scoped_electrode = power_df

        for band in bands:
            test_specific_data = scoped_electrode[(scoped_electrode.band == band)]
            test_group = test_specific_data[test_specific_data.participantID.isin(participants)]
            for dataset in ['ml', 'beta', 'pink']:
                test_group_dataset_scoped = test_group[test_group.dataset_name == dataset]
                values = {
                    'group': 'test',
                    'dataset': dataset,
                    'band': band,
                    'mean_start': test_group_dataset_scoped[['start']].mean().values[0],
                    'mean_end': test_group_dataset_scoped[['end']].mean().values[0],
                    'std_start': test_group_dataset_scoped[['start']].std().values[0],
                    'std_end': test_group_dataset_scoped[['end']].std().values[0],
                    'mean_abs_diff': test_group_dataset_scoped[['overall']].abs().mean().values[0],
                    'std_abs_diff': test_group_dataset_scoped[['overall']].abs().std().values[0],
                    'region': region
                }
                statistics_across = statistics_across.append(values, ignore_index=True)
            control_group = test_specific_data[test_specific_data.participantID.isin(control) &
                                               (test_specific_data.band == band)]
            control_group_dataset_scoped = control_group[control_group.dataset_name == 'pink']

            values = {
                'group': 'control',
                'dataset': dataset,
                'band': band,
                'mean_start': control_group_dataset_scoped[['start']].mean().values[0],
                'mean_end': control_group_dataset_scoped[['end']].mean().values[0],
                'std_start': control_group_dataset_scoped[['start']].std().values[0],
                'std_end': control_group_dataset_scoped[['end']].std().values[0],
                'mean_abs_diff': control_group_dataset_scoped[['overall']].abs().mean().values[0],
                'std_abs_diff': control_group_dataset_scoped[['overall']].abs().std().values[0],
                'region': region
            }
            statistics_across = statistics_across.append(values, ignore_index=True)

    for band in bands:
        band_filtered = increasing_electrodes[increasing_electrodes.band == band]
        band_filtered = band_filtered.sort_values(by=['dataset', 'group'])
        band_filtered.to_csv(f"meta_analysis/{band}_increasingCount_{modality}", index=False)

        statistics_across_band_filtered = statistics_across[statistics_across.band == band]
        statistics_across_band_filtered = statistics_across_band_filtered.sort_values(by=['dataset', 'group'])
        statistics_across_band_filtered.to_csv(f"meta_analysis/{band}_stats_{modality}", index=False)

        power_change_columns = power_df[
            ['participantID', 'dataset_name', 'region', 'band', 'group', 'start_three', 'start_six', 'start_nine',
             'start_twelve', 'three_six', 'six_nine', 'nine_twelve', 'twelve_fifteen', 'overall', 'increased_overall',
             'increased_start_three', 'increased_start_six', 'increased_start_nine',
             'increased_start_twelve', 'increased_three_six', 'increased_six_nine',
             'increased_nine_twelve', 'increased_twelve_fifteen']]
        power_change_filtered = power_change_columns[power_change_columns.band == band]
        power_change_sorted = power_change_filtered.sort_values(by=['dataset_name', 'group'])
        power_change_sorted.to_csv(f"meta_analysis/{band}_delta_{modality}", index=False)

        meta_increasing_band = meta_increasing[meta_increasing.band == band]
        meta_increasing_band_sorted = meta_increasing_band.sort_values(by=['dataset', 'group'])
        meta_increasing_band_sorted.to_csv(f"meta_analysis/{band}_metaIncreasing_{modality}", index=False)


def analyse_percentage_change(modality):
    if modality == 'power':
        filename = percentage_power_analysis_file
    else:
        filename = percentage_coherence_analysis_file
    percentage_change_df = pd.read_csv(filename, skipinitialspace=True)
    percentage_change_df = percentage_change_df.sort_values(by=['dataset', 'band'])
    if 'group' not in percentage_change_df.keys():
        percentage_change_df['group'] = percentage_change_df['Participant'].apply(
            lambda x: 'test' if x in test_participants
            else 'control')

    regions = ['T', 'TP', 'FT', 'all', 'T_and_TP']
    datasets = ['ml', 'beta', 'pink']
    bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
    averages = pd.DataFrame(columns=['dataset', 'band', 'group', 'region', 'average', 'Q1', 'Q3', 'median', 'max', 'IQR'
        , 'outlier_upper_bound', 'outlier_lower_bound'])

    for region in regions:
        for band in bands:
            for dataset in datasets:
                if dataset == 'pink':
                    groups = ['test', 'control', 'all']
                else:
                    groups = ['test']
                for group in groups:
                    if group == 'all':
                        percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                                       (percentage_change_df.dataset == dataset) &
                                                                       (percentage_change_df.Participant.isin(
                                                                           participants_to_include))]
                    else:
                        percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                                       (percentage_change_df.dataset == dataset) &
                                                                       (percentage_change_df.Participant.isin(
                                                                           participants_to_include)) &
                                                                       (percentage_change_df.group == group)]

                    if region == 'all':
                        percentage_region_scoped = percentage_above_scoped[
                            (percentage_above_scoped.region == 'average')]
                        # percentage_region_scoped = percentage_above_scoped
                    elif region == 'T_and_TP':
                        percentage_region_scoped = percentage_above_scoped[(percentage_above_scoped.region.isin(
                            ['T', 'TP']))]
                    else:
                        percentage_region_scoped = percentage_above_scoped[(percentage_above_scoped.region == region)]
                    q1 = np.percentile(percentage_region_scoped['average'], 25)
                    q3 = np.percentile(percentage_region_scoped['average'], 75)
                    iqr = np.subtract(*np.percentile(percentage_region_scoped['average'], [75, 25]))
                    result = {'dataset': dataset, 'band': band, 'group': group, 'region': region,
                              'average': percentage_region_scoped['average'].mean(),
                              'max': percentage_region_scoped['average'].max(),
                              'Q1': q1,
                              'Q3': q3,
                              'median': np.percentile(percentage_region_scoped['average'], 50),
                              'IQR': iqr,
                              'outlier_upper_bound': q3 + 1.5 * iqr,
                              'outlier_lower_bound': q1 - 1.5 * iqr}
                    averages = averages.append(result, ignore_index=True)

    percentage_change_df.to_csv(filename, index=False)
    for band in bands:
        averages_band_scoped = averages[averages.band == band]
        averages_band_sorted = averages_band_scoped.sort_values(by=['dataset', 'group'])
        averages_band_sorted.to_csv(f"meta_analysis/{band}_percentage_increase_{modality}", index=False)


def t_test(modality):
    if modality == 'power':
        # filename = power_analysis_file
        filename = percentage_power_analysis_file
    else:
        filename = percentage_coherence_analysis_file

    percentage_change_df = pd.read_csv(filename, skipinitialspace=True)
    percentage_change_df = percentage_change_df.sort_values(by=['dataset', 'band'])
    if 'group' not in percentage_change_df.keys():
        percentage_change_df['group'] = percentage_change_df['Participant'].apply(
            lambda x: 'test' if x in test_participants
            else 'control')

    regions = ['T', 'TP', 'FT', 'all', 'T_and_TP']
    datasets = ['ml']
    datasets_compare = ['ml', 'pink', 'beta']
    # SWAP Below to do normal analysis. TODO make swap dynamic
    # datasets = ['beta', 'ml', 'pink']
    # datasets_compare = ['ml', 'pink']
    bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
    averages = pd.DataFrame(columns=['dataset', 'dataset compare', 'band', 'group compare', 'region', 't', 'p'])
    percentage_change_df = pd.concat([percentage_change_df], ignore_index=True)

    for region in regions:
        for band in bands:
            for dataset in datasets:
                if dataset == 'pink':
                    groups = ['test', 'control', 'all']
                else:
                    groups = ['test']
                for compare_ds in datasets_compare:
                    if compare_ds == 'pink':
                        groups_compare = ['test', 'control', 'all']
                        # groups_compare = ['control']
                    else:
                        groups_compare = ['test']
                    for group in groups_compare:
                        if dataset == compare_ds and group == 'test':
                            continue
                        if group == 'all':
                            percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                                           (percentage_change_df.dataset == dataset) &
                                                                           (percentage_change_df.Participant.isin(
                                                                               p_exploit))]
                            # SWAP Below to do normal analysis. TODO make swap dynamic
                            # percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                            #                                                (percentage_change_df.dataset == dataset) &
                            #                                                (percentage_change_df.Participant.isin(
                            #                                                    participants_to_include))]
                            percentage_above_scoped_compare = percentage_change_df[(percentage_change_df.band == band) &
                                                                                   (
                                                                                           percentage_change_df.dataset == compare_ds) &
                                                                                   (
                                                                                       percentage_change_df.Participant.isin(
                                                                                           participants_to_include))]

                        else:
                            percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                                           (percentage_change_df.dataset == dataset) &
                                                                           (percentage_change_df.Participant.isin(
                                                                               p_exploit)) &
                                                                           (percentage_change_df.group == 'test')]
                            # SWAP Below to do normal analysis. TODO make swap dynamic
                            # percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                            #                                                (percentage_change_df.dataset == dataset) &
                            #                                                (percentage_change_df.Participant.isin(
                            #                                                    participants_to_include)) &
                            #                                                (percentage_change_df.group == 'test')]
                            percentage_above_scoped_compare = percentage_change_df[(percentage_change_df.band == band) &
                                                                                   (
                                                                                           percentage_change_df.dataset == compare_ds) &
                                                                                   (
                                                                                       percentage_change_df.Participant.isin(
                                                                                           participants_to_include)) &
                                                                                   (
                                                                                           percentage_change_df.group == group)]

                        if region == 'all':
                            percentage_region_scoped = percentage_above_scoped[
                                (percentage_above_scoped.region == 'average')]
                            percentage_region_scoped_compare = percentage_above_scoped_compare[
                                (percentage_above_scoped_compare.region == 'average')]
                            # percentage_region_scoped = percentage_above_scoped
                        elif region == 'T_and_TP':
                            percentage_region_scoped = percentage_above_scoped[(percentage_above_scoped.region.isin(
                                ['T', 'TP']))]
                            percentage_region_scoped_compare = percentage_above_scoped_compare[
                                (percentage_above_scoped_compare.region.isin(
                                    ['T', 'TP']))]
                        else:
                            percentage_region_scoped = percentage_above_scoped[
                                (percentage_above_scoped.region == region)]
                            percentage_region_scoped_compare = percentage_above_scoped_compare[
                                (percentage_above_scoped_compare.region == region)]

                        # print(percentage_region_scoped_compare[['3', '6', '9', '12', '15', 'average']].values.flatten())
                        t, p = ttest_ind(percentage_region_scoped[['3', '6', '9', '12', '15', 'average']].values.
                                         flatten(),
                                         percentage_region_scoped_compare[
                                             ['3', '6', '9', '12', '15', 'average']].values.
                                         flatten(), equal_var=False, permutations=300)
                        # alternative = 'greater'
                        # t, p = ttest_ind_from_stats(mean1=percentage_region_scoped['%above'].mean(), std1=
                        #                             np.std(percentage_region_scoped['%above']),
                        #                             nobs1=len(percentage_region_scoped['%above']),
                        #                             mean2=percentage_region_scoped_compare['%above'].mean(),
                        #                             std2=np.std(percentage_region_scoped_compare['%above']),
                        #                             nobs2=len(percentage_region_scoped_compare['%above']),
                        #                             equal_var=False, alternative='greater')
                        result = {'dataset': dataset, 'dataset compare': compare_ds, 'band': band, 'group compare':
                            group, 'region': region,
                                  't': t,
                                  'p': p}
                        # print(result)
                        averages = averages.append(result, ignore_index=True)

    # percentage_change_df.to_csv(filename, index=False)
    for band in bands:
        averages_band_scoped = averages[averages.band == band]
        averages_band_sorted = averages_band_scoped.sort_values(by=['dataset', 'group compare', 'dataset compare'])
        # SWAP Below to do normal analysis. TODO make swap dynamic
        # averages_band_sorted.to_csv(f"meta_analysis/{band}_percentage_increase_t_test_{modality}", index=False)
        averages_band_sorted.to_csv(f"meta_analysis/{band}_T_V_{modality}", index=False)


def t_test_power_tmp(modality):
    if modality == 'power':
        filename = power_analysis_file
        # filename = percentage_power_analysis_file
    else:
        filename = percentage_coherence_analysis_file

    percentage_change_df = pd.read_csv(filename, skipinitialspace=True)
    # percentage_change_df = percentage_change_df.sort_values(by=['dataset', 'band'])
    percentage_change_df = percentage_change_df.sort_values(by=['dataset_name', 'band'])
    if 'group' not in percentage_change_df.keys():
        percentage_change_df['group'] = percentage_change_df['participantID'].apply(
            lambda x: 'test' if x in test_participants
            else 'control')
    # TODO Increase this by expanding to other diff values to justify statistical significance
    percentage_change_df['overall'] = percentage_change_df['end'] - percentage_change_df['start']
    percentage_change_df['overall'] = percentage_change_df['overall'].abs()
    percentage_change_df['start_three'] = percentage_change_df['three_min'] - percentage_change_df['start']
    percentage_change_df['start_three'] = percentage_change_df['start_three'].abs()
    percentage_change_df['start_six'] = percentage_change_df['six_min'] - percentage_change_df['start']
    percentage_change_df['start_six'] = percentage_change_df['start_six'].abs()
    percentage_change_df['start_nine'] = percentage_change_df['nine_min'] - percentage_change_df['start']
    percentage_change_df['start_nine'] = percentage_change_df['start_nine'].abs()
    percentage_change_df['start_twelve'] = percentage_change_df['twelve_min'] - percentage_change_df['start']
    percentage_change_df['start_twelve'] = percentage_change_df['start_twelve'].abs()
    percentage_change_df['increased_overall'] = percentage_change_df['overall'].apply(lambda x: 2 if x > 0 else 1)
    # percentage_change_df = pd.concat([percentage_change_df] * 5, ignore_index=True)

    regions = ['T', 'TP', 'FT', 'T_and_TP']
    datasets = ['beta', 'ml', 'pink']
    datasets_compare = ['ml', 'pink']
    bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
    averages = pd.DataFrame(columns=['dataset', 'dataset compare', 'band', 'group compare', 'region', 't', 'p'])

    for region in regions:
        for band in bands:
            for dataset in datasets:
                if dataset == 'pink':
                    groups = ['test', 'control', 'all']
                else:
                    groups = ['test']
                for compare_ds in datasets_compare:
                    if compare_ds == 'pink':
                        groups_compare = ['test', 'control', 'all']
                    else:
                        groups_compare = ['test']
                    for group in groups_compare:
                        if group == 'all':
                            percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                                           (
                                                                                   percentage_change_df.dataset_name == dataset) &
                                                                           (percentage_change_df.participantID.isin(
                                                                               participants_to_include))]
                            percentage_above_scoped_compare = percentage_change_df[(percentage_change_df.band == band) &
                                                                                   (
                                                                                           percentage_change_df.dataset_name == compare_ds) &
                                                                                   (
                                                                                       percentage_change_df.participantID.isin(
                                                                                           participants_to_include))]
                        else:
                            percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                                           (
                                                                                   percentage_change_df.dataset_name == dataset) &
                                                                           (percentage_change_df.participantID.isin(
                                                                               participants_to_include)) &
                                                                           (percentage_change_df.group == 'test')]
                            percentage_above_scoped_compare = percentage_change_df[(percentage_change_df.band == band) &
                                                                                   (
                                                                                           percentage_change_df.dataset_name == compare_ds) &
                                                                                   (
                                                                                       percentage_change_df.participantID.isin(
                                                                                           participants_to_include)) &
                                                                                   (
                                                                                           percentage_change_df.group == group)]

                        if region == 'all':
                            percentage_region_scoped = percentage_above_scoped[
                                (percentage_above_scoped.region == 'average')]
                            percentage_region_scoped_compare = percentage_above_scoped_compare[
                                (percentage_above_scoped_compare.region == 'average')]
                            # percentage_region_scoped = percentage_above_scoped
                        elif region == 'T_and_TP':
                            percentage_region_scoped = percentage_above_scoped[(percentage_above_scoped.region.isin(
                                ['T', 'TP']))]
                            percentage_region_scoped_compare = percentage_above_scoped_compare[
                                (percentage_above_scoped_compare.region.isin(
                                    ['T', 'TP']))]
                        else:
                            percentage_region_scoped = percentage_above_scoped[
                                (percentage_above_scoped.region == region)]
                            percentage_region_scoped_compare = percentage_above_scoped_compare[
                                (percentage_above_scoped_compare.region == region)]

                        t, p = ttest_ind(percentage_region_scoped[['overall', 'start_three', 'start_six', 'start_nine',
                                                                   'start_twelve']].values.
                                         flatten(),
                                         percentage_region_scoped_compare[
                                             ['overall', 'start_three', 'start_six', 'start_nine', 'start_twelve']].
                                         values.flatten(), equal_var=False, permutations=500, alternative='greater')
                        # t, p = ttest_ind(percentage_region_scoped['%above'],
                        #                  percentage_region_scoped_compare['%above'], equal_var=False,
                        #                  permutations=500, alternative='greater')
                        # alternative = 'greater'
                        # t, p = ttest_ind_from_stats(mean1=percentage_region_scoped['%above'].mean(), std1=
                        #                             np.std(percentage_region_scoped['%above']),
                        #                             nobs1=len(percentage_region_scoped['%above']),
                        #                             mean2=percentage_region_scoped_compare['%above'].mean(),
                        #                             std2=np.std(percentage_region_scoped_compare['%above']),
                        #                             nobs2=len(percentage_region_scoped_compare['%above']),
                        #                             equal_var=False, alternative='greater')
                        result = {'dataset': dataset, 'dataset compare': compare_ds, 'band': band, 'group compare':
                            group, 'region': region,
                                  't': t,
                                  'p': p}
                        averages = averages.append(result, ignore_index=True)

    # percentage_change_df.to_csv(filename, index=False)
    for band in bands:
        averages_band_scoped = averages[averages.band == band]
        averages_band_sorted = averages_band_scoped.sort_values(by=['dataset', 'group compare', 'dataset compare'])
        # averages_band_sorted.to_csv(f"meta_analysis/{band}_percentage_increase_t_test_{modality}", index=False)
        averages_band_sorted.to_csv(f"meta_analysis/{band}_overall_t_test_{modality}", index=False)


analyse_results(modality='coherence')

# analyse_percentage_change(modality='power')
#  Currently doing ML vs non ml t test T_V ... csv files
# t_test(modality='power')
# test()
# Does t test for overall diff overall_t_test_...
# t_test_power_tmp(modality='power')

# record_action_order()

# k_means_analysis()
