from analysis import Analysis
import h5py
import time
from participant_info import test_Q_datapoint

a = Analysis()

hf = h5py.File(f'../data/custom_suite/{test_Q_datapoint}/b_pink.h5', 'r')

samples = hf['raw_data']
eeg_data = samples[()]
eeg_data = eeg_data[512*60:512*60+512 * 60*3,:]

tic = time.perf_counter()
a.get_features_and_reward(eeg_data)
print(len(eeg_data))
toc = time.perf_counter()
print(f"Data handling thread tood: {toc - tic:0.4f} seconds")