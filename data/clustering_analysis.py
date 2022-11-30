from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from data.constants import power_analysis_file, coherence_analysis_file, participants, test_participants,\
    control_participants
from participant_info import test


def k_means_analysis():
    power_data = pd.read_csv(power_analysis_file, skipinitialspace=True)
    power_data = power_data.drop(power_data[power_data.participantID == 'Jasp'].index)
    # Normalise
    power_data['end'] = power_data['end']/power_data['start']
    # power_data['start'] = power_data['start']/power_data['start'].max()

    coherence = pd.read_csv(coherence_analysis_file, skipinitialspace=True)
    coherence = coherence.drop(coherence[coherence.participantID == 'Jasp'].index)
    # Normalise
    coherence['end'] = coherence['end']/coherence['end'].max()
    coherence['start'] = coherence['start']/coherence['start'].max()

    # power_bands = ['beta', 'alpha', 'beta_entrain', 'beta_entrain_low', 'theta']
    power_bands = ['theta']
    coherence_bands = ['beta_entrain', 'beta_entrain_low']
    # datasets =
    # regions = ['T', 'TP', 'FT', 'all']
    for band in power_bands:
        # power_data_band_scoped = power_data[(power_data.band == band) & (power_data.region.isin(['T', 'TP']))]
        # coherence_band_scoped = coherence[(coherence.band == band) & (coherence.region.isin(['T', 'TP']))]

        power_data_band_scoped = power_data[(power_data.band == band)]
        coherence_band_scoped = coherence[(coherence.band == band)]
        power_data_band_scoped.set_index('participantID')
        coherence_band_scoped.set_index('participantID')
        # & (power_data.region.isin(['T', 'TP'])
        # print(power_data_band_scoped)
        # for region in regions:
        # region_scoped = power_data_band_scoped[(power_data_band_scoped.region == region)]

        region_aggregated = power_data_band_scoped.groupby(['participantID', 'dataset_name']).agg(lambda x: x.tolist())
        region_aggregated_coherence = coherence_band_scoped.groupby(['participantID', 'dataset_name']).agg(lambda x: x.tolist())

        power_data_initial = region_aggregated[['end']]
        power_data_initial_coherence = region_aggregated_coherence[['end']]

        # print(power_data_initial['start'])
        # print(power_data_initial_coherence['start'])
        # print(power_data_initial['start'].tolist())
        # print(power_data_initial_coherence['start'].tolist())


        coherence_data = power_data_initial_coherence['end'].tolist()

        data = [[]] * len(power_data)

        power_data = power_data_initial['end'].tolist()
        # print(power_data)

        # for idx, power_data_values in enumerate(power_data):
        #
            # data[idx] = np.hstack((power_data, coherence_data[idx]))
        data = power_data
        params = {'n_clusters': 3, 'random_state': 0, 'n_init': 50, 'max_iter': 1000, 'init': 'random'}
        kmeans = KMeans(**params).fit(data)
        labels = kmeans.labels_
        # # labels = kmeans.predict(data)
        # print(labels)
        # labels = kmeans.predict(data)
        # print(labels)
        # print()
        # print(power_data_initial.index.values)
        # TODO Loop through time periods here
        # power_data_start = region_aggregated[['start']]
        # initial_power_data = power_data_start['start'].tolist()
        # labels_start = kmeans.predict(initial_power_data)


        results_clustering = pd.DataFrame(columns=['ID', 'dataset', 'cluster_initial', 'cluster_end', 'group'])
        ids = power_data_initial.index.values
        for idx, label in enumerate(labels):
            if ids[idx][0] in test_participants:
                group = 'test'
            else:
                group = 'control'
            entry = {
                'ID': ids[idx][0],
                'group': group,
                'dataset': ids[idx][1],
                # 'cluster_initial': labels_start[idx],
                'cluster_end': label
            }
            results_clustering = results_clustering.append(entry, ignore_index=True)
        results_clustering = results_clustering.sort_values(by=['dataset', 'group'])
        print(results_clustering)
        print(kmeans.cluster_centers_)
        print(kmeans.inertia_)
        print(kmeans.n_iter_)

        results_clustering.to_csv(f"meta_analysis/{band}_clusterAnalysis", index=False)
        # print(kmeans.predict(power_data_initial['start'].tolist()))








    # X['Cluster'] = model.predict(X)
    # X['Survived'] = y
    # mapper = X.groupby('Cluster')['Survived'].mean().round().to_dict()

    # df['Survived'] = model.predict(X_test)
    # overall_predictions = df.Survived.map(mapper).astype(np.int)
    #
    # input_predictions = X['Cluster'].map(mapper).astype(np.int)
    #
    # accuracy = accuracy_score(y, input_predictions)