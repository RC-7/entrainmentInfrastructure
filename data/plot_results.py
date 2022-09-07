import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

datasets = ['ml', 'beta', 'pink']
groups = ['test', 'control', 'all']
regions = ['T', 'TP', 'FT', 'all']

bands = ['beta', 'alpha', 'theta']

test = ['V', 'A', 'S', 'D', 'J', 'T']
control = ['B', 'El', 'Zo', 'H', 'P', 'St']


def plot_time_count_changes():
    for band in bands:
        filename = f'meta_analysis/{band}_increasingCount_power'
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
        filename_save = f'figures/time_changes_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()


def plot_mean_abs_diff():
    for band in bands:
        filename = f'meta_analysis/{band}_stats_power'
        df = pd.read_csv(filename)
        df = df[(df.region != 'all')]
        df["recording"] = df['group'].astype(str) + "  group, " + df["dataset"] + " stimulus"
        df["recording"] = df["recording"].str.replace('pink', 'control')
        df['mean absolute difference'] = df['mean_abs_diff']
        sns.barplot(data=df, x="region", y='mean absolute difference', hue='recording')
        filename_save = f'figures/mean_abs_diff_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()



def plot_mean_increasing():
    for band in bands:
        filename = f'meta_analysis/{band}_metaIncreasing_power'
        df = pd.read_csv(filename)
        df = df[(df.metric == 'increased_overall') & (df.group != 'all')]
        # df["test"] = np.where(df["dataset"] == "pink", f'{df["group"]}, control', f'{df["group"]}, {df["dataset"]}')
        df["recording"] = df['group'].astype(str) + "  group,\n" + df["dataset"] + " stimulus"
        df["recording"] = df["recording"].str.replace('pink', 'control')
        df["mean increasing count"] = df['mean']
        sns.barplot(data=df, x="recording",  y='mean increasing count')
        filename_save = f'figures/mean_increase_count_{band}.pdf'
        plt.tight_layout()
        plt.savefig(filename_save)
        plt.close()

def plot_increase_count_buckets():
    for band in bands:
        filename = f'meta_analysis/{band}_metaIncreasing_power'
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
        filename_save = f'figures/increaseCountBucket_{band}.pdf'
        plt.savefig(filename_save)
        plt.close()


def main():
    # plot_increase_count_buckets()
    # plot_mean_increasing()
    # plot_mean_abs_diff()
    plot_time_count_changes()

if __name__ == '__main__':
    main()
