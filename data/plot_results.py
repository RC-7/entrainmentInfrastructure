import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from data.constants import percentage_power_analysis_file, percentage_coherence_analysis_file, test_participants

datasets = ['ml', 'beta', 'pink']
groups = ['test', 'control', 'all']
regions = ['T', 'TP', 'FT', 'all']

bands = ['beta', 'alpha', 'theta']

test = ['V', 'A', 'S', 'D', 'J', 'T']
control = ['B', 'El', 'Zo', 'H', 'P', 'St']
regions = ['T', 'TP', 'FT']


def std_div_plot():
    for band in bands:
        filename = f'meta_analysis/{band}_stats_power'
        df = pd.read_csv(filename)
        df = df[(df.region != 'all')]
        df["recording"] = df['group'].astype(str) + "  group " + df["dataset"] + " stimulus\n"
        df["recording"] = df["recording"].str.replace('pink', 'control')
        df['start'] = df['std_start']
        df['end'] = df['std_end']
        df = df[['recording', 'start', 'end', 'region']]
        ax = df.set_index(['recording', 'region']).plot( kind='barh', stacked=True)
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels = [s.replace("(", "").replace(')', "") + ' region' for s in labels]
        ax.set_yticklabels(labels)
        plt.xlabel('standard deviation')
        filename_save = f'figures/std_div_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def t_test_percentage(file_suffix='_percentage_increase_t_test_power'):
    for band in bands:
        filename = f'meta_analysis/{band}_{file_suffix}'
        df = pd.read_csv(filename)
        df["comparison"] = df['dataset'].astype(str) + " stimulus compared to \n " + df['group compare'].astype(str) +\
                           "  group, " + df["dataset compare"] + " stimulus"
        df["comparison"] = df["comparison"].str.replace('pink', 'control')
        datasets = ['beta', 'ml']
        df = df[(df['group compare'] != 'all') & (df['dataset'] != 'pink')]
        # for ds in datasets:
        scoped_df = df[(df.region.isin(
            regions))]
        graph = sns.barplot(data=scoped_df, x='p', y='comparison', hue='region')
        graph.axvline(0.05)
        # graph.axhline(0.08, color='red')
        plt.tight_layout()
        filename_save = f'figures/{file_suffix}_{band}.pdf'
        plt.savefig(filename_save)
        plt.close()


def boxplot_percentage_comparative(modality='power'):
    if modality == 'power':
        filename = percentage_power_analysis_file
    else:
        filename = percentage_coherence_analysis_file

    exploit = ['V', 'T']
    percentage_change_df = pd.read_csv(filename, skipinitialspace=True)
    percentage_change_df = percentage_change_df.sort_values(by=['dataset', 'band'])
    percentage_change_df = percentage_change_df[(percentage_change_df.dataset == 'ml')]
    percentage_change_df["recording"] = np.where(percentage_change_df['Participant'].isin(exploit), 'exploit', 'explore')
    # percentage_change_df["recording"] = percentage_change_df["recording"].str.replace('pink', 'control')
    if 'group' not in percentage_change_df.keys():
        percentage_change_df['group'] = percentage_change_df['Participant'].apply(
            lambda x: 'test' if x in test_participants
            else 'control')

    regions = ['T', 'TP', 'FT']

    participants_to_include = ['T', 'V', 'St', 'J', 'D', 'El', 'P',
                               'H', 'Zo', 'B', 'S', 'A']
    for band in bands:
        percentage_change_df['% above initial'] = percentage_change_df['average']
        percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                       (percentage_change_df.Participant.isin(
                                                           participants_to_include)) & (
                                                       percentage_change_df.region.isin(
                                                           regions))]
        sns.boxplot(data=percentage_above_scoped, x='recording', y='% above initial', hue='region',
                    medianprops=dict(color="red", alpha=0.7), showmeans=True)
        filename_save = f'figures/%change_mlcomp_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def boxplot_percentage(modality='power'):
    if modality == 'power':
        filename = percentage_power_analysis_file
        value_key = 'average'
    else:
        filename = percentage_coherence_analysis_file
        value_key = '%above'
        bands = ['beta_entrain', 'beta_entrain_low']
    percentage_change_df = pd.read_csv(filename, skipinitialspace=True)
    percentage_change_df = percentage_change_df.sort_values(by=['dataset', 'band'])
    percentage_change_df["recording"] = percentage_change_df['group'].astype(str) + "  group\n, " + percentage_change_df[
        "dataset"] + " stimulus"
    percentage_change_df["recording"] = percentage_change_df["recording"].str.replace('pink', 'control')
    if 'group' not in percentage_change_df.keys():
        percentage_change_df['group'] = percentage_change_df['Participant'].apply(
            lambda x: 'test' if x in test_participants
            else 'control')

    regions = ['T', 'TP', 'FT']
    participants_to_include = ['T', 'V', 'St', 'J', 'D', 'El', 'P',
                               'H', 'Zo', 'B', 'S', 'A']
    for band in bands:
        percentage_change_df['% above initial'] = percentage_change_df[value_key]
        percentage_above_scoped = percentage_change_df[(percentage_change_df.band == band) &
                                                       (percentage_change_df.Participant.isin(
                                                           participants_to_include)) & (percentage_change_df.region.isin(
            regions))]
        sns.boxplot(data=percentage_above_scoped, x='recording', y='% above initial', hue='region',
                    medianprops=dict(color="red", alpha=0.7), showmeans=True)
        filename_save = f'figures/percentage_change_{modality}_bw_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def plot_time_count_changes(modality='power'):
    if modality == 'coherence':
        bands = ['beta_entrain', 'beta_entrain_low']
    for band in bands:
        # filename = f'meta_analysis/{band}_increasingCount_power'
        if modality == 'power':
            filename = f'meta_analysis/{band}_increasingCount_power'
        else:
            filename = f'meta_analysis/{band}_increasingCount_coherence'
        df = pd.read_csv(filename)
        df["recording"] = df['group'].astype(str) + "  group, " + df["dataset"] + " stimulus"
        df["recording"] = df["recording"].str.replace('pink', 'control')
        collapsing_columns = ['increased_start_three', 'increased_start_six', 'increased_start_nine',
                              'increased_start_twelve', 'increased_overall']
        unique_recordings = df['recording'].unique()
        colours = ['green', 'blue', 'red', 'orange']
        colour_iter = 0
        for rec in unique_recordings:
            ds_scoped = df[(df['recording'] == rec)]
            values = []
            for c in collapsing_columns:
                values.append(ds_scoped[[c]].mean()[0])
            time = [3, 6, 9, 12, 15]
            plt.plot(time, values, color=colours[colour_iter])
            colour_iter += 1
        plt.legend(unique_recordings)
        plt.xlabel('time evaluated')
        plt.ylabel('mean number of increasing electrodes')
        filename_save = f'figures/time_changes_{modality}_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def plot_mean_abs_diff(modality='power'):
    if modality == 'coherence':
        bands = ['beta_entrain', 'beta_entrain_low']
    for band in bands:
        if modality == 'power':
            filename = f'meta_analysis/{band}_stats_power'
        else:
            filename = f'meta_analysis/{band}_stats_coherence'
        df = pd.read_csv(filename)
        df = df[(df.region != 'all')]
        df["recording"] = df['group'].astype(str) + "  group, " + df["dataset"] + " stimulus"
        df["recording"] = df["recording"].str.replace('pink', 'control')
        df['mean absolute difference'] = df['mean_abs_diff']
        sns.barplot(data=df, x="region", y='mean absolute difference', hue='recording')
        filename_save = f'figures/mean_abs_diff_{modality}_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def plot_mean_increasing(modality='power'):
    if modality == 'coherence':
        bands = ['beta_entrain', 'beta_entrain_low']
    for band in bands:
        if modality == 'power':
            filename = f'meta_analysis/{band}_metaIncreasing_power'
        else:
            filename = f'meta_analysis/{band}_metaIncreasing_coherence'
        df = pd.read_csv(filename)
        df = df[(df.metric == 'increased_overall') & (df.group != 'all')]
        # df["test"] = np.where(df["dataset"] == "pink", f'{df["group"]}, control', f'{df["group"]}, {df["dataset"]}')
        df["recording"] = df['group'].astype(str) + "  group,\n" + df["dataset"] + " stimulus"
        df["recording"] = df["recording"].str.replace('pink', 'control')
        df["mean increasing count"] = df['mean']
        sns.barplot(data=df, x="recording", y='mean increasing count')
        filename_save = f'figures/mean_increase_count_{modality}_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def plot_increase_count_buckets(modality='power'):
    if modality == 'coherence':
        bands = ['beta_entrain', 'beta_entrain_low']
    for band in bands:
        if modality == 'power':
            filename = f'meta_analysis/{band}_metaIncreasing_power'
        else:
            filename = f'meta_analysis/{band}_metaIncreasing_coherence'
        df = pd.read_csv(filename)

        df = df[(df.metric == 'increased_overall') & (df.group != 'all')]
        new_format_columns = ['metric', 'number_count', 'group', 'dataset']
        new_format = pd.DataFrame(columns=new_format_columns)
        collapsing_columns = ['0_count', '1_count', '2_count', '3_count']
        for index, row in df.iterrows():
            for c in collapsing_columns:
                if row["dataset"] == 'ml':
                    stimulus = 'ML'
                elif row["dataset"] == 'pink':
                    stimulus = 'control'
                else:
                    stimulus = row["dataset"]
                to_write = {
                    'Increasing count bucket': c.split("_")[0],
                    'Count': row[c],
                    'group': row['group'],
                    'dataset': row['dataset'],
                    'key': f'{row["group"]} group, {stimulus} stimulus'
                }
                new_format = new_format.append(to_write, ignore_index=True)

        bar_plot = sns.barplot(data=new_format, x="Increasing count bucket", hue="key", y='Count')
        # fig = bar_plot.get_figure()
        filename_save = f'figures/increaseCountBucket_{modality}_{band}.pdf'
        plt.savefig(filename_save)
        plt.close()


def plot_power_values():
    # ds_name = 'beta_audio'
    # ds_names = ['ml_beta_audio', 'beta_audio', 'pink_audio']
    ds_names = ['pink_audio']
    for ds_name in ds_names:
        for band in bands:
            fig, ax = plt.subplots()
            for region in regions:
                if ds_name != 'pink_audio':
                    np_ds_filename_data = f'stft_region_averaged/ft_values/V_{ds_name}_{band}_filtered_Power_{region}.npy'
                else:
                    np_ds_filename_data = f'stft_region_averaged/ft_values/J_{ds_name}_{band}_filtered_Power_{region}.npy'
                power_values = np.load(np_ds_filename_data, allow_pickle=True)[1]
                time = np.load(np_ds_filename_data, allow_pickle=True)[0]
                ax.plot(time, power_values)
            ax.legend(regions)
            filename_save = f'figures/test_participant_raw_power_values_{band}_{ds_name}.pdf'
            plt.savefig(filename_save)
            plt.close()
            plt.xlabel('time (min)')
            plt.ylabel('power (dB)')

    ds_names = ['pink_audio']
    for ds_name in ds_names:
        for band in bands:
            fig, ax = plt.subplots()
            for region in regions:
                np_ds_filename_data = f'stft_region_averaged/ft_values/Zo_{ds_name}_{band}_filtered_Power_{region}.npy'
                power_values = np.load(np_ds_filename_data, allow_pickle=True)[1]
                time = np.load(np_ds_filename_data, allow_pickle=True)[0]
                ax.plot(time, power_values)
            ax.legend(regions)
            filename_save = f'figures/control_participant_raw_power_values_{band}_{ds_name}.pdf'
            plt.savefig(filename_save)
            plt.close()
            plt.xlabel('time (min)')
            plt.ylabel('power (dB)')


def plot_coherence_values():
    # ds_name = 'beta_audio'
    ds_names = [ 'pink_audio']
    bands = ['beta_entrain', 'beta_entrain_low']
    for ds_name in ds_names:
        for band in bands:
            fig, ax = plt.subplots()
            for region in regions:
                np_ds_filename_data = f'Clustering/A_{ds_name}_{band}_Clustering_{region}.npy'
                power_values = np.load(np_ds_filename_data, allow_pickle=True)[1]
                time = np.load(np_ds_filename_data, allow_pickle=True)[0]
                ax.plot(time, power_values)
            ax.legend(regions)
            filename_save = f'figures/test_participant_raw_clustering_{band}_{ds_name}_new.pdf'
            plt.savefig(filename_save)
            plt.close()
            plt.xlabel('time (min)')
            plt.ylabel('clustering coefficient')

    # ds_names = ['pink_audio']
    # for ds_name in ds_names:
    #     for band in bands:
    #         fig, ax = plt.subplots()
    #         for region in regions:
    #             np_ds_filename_data = f'Clustering/El_{ds_name}_{band}_Clustering_{region}.npy'
    #             power_values = np.load(np_ds_filename_data, allow_pickle=True)[1]
    #             time = np.load(np_ds_filename_data, allow_pickle=True)[0]
    #             ax.plot(time, power_values)
    #         ax.legend(regions)
    #         filename_save = f'figures/control_participant_clusterings_{band}_{ds_name}.pdf'
    #         plt.savefig(filename_save)
    #         plt.close()
    #         plt.xlabel('time (min)')
    #         plt.ylabel('clustering coefficient')


def main():
    # plot_increase_count_buckets(modality='coherence')
    # plot_mean_increasing(modality='coherence')
    # plot_mean_abs_diff(modality='coherence')
    # plot_time_count_changes(modality='coherence')
    # boxplot_percentage(modality='coherence')
    # t_test_percentage(file_suffix='overall_t_test_power')
    # std_div_plot()
    # plot_power_values()
    plot_coherence_values()
    # boxplot_percentage_comparative()


if __name__ == '__main__':
    main()
