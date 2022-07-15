from analysis import Analysis
import h5py
import time

a = Analysis()

hf = h5py.File('../data/custom_suite/Full_run_S/b_pink.h5', 'r')

samples = hf['raw_data']
eeg_data = samples[()]
eeg_data = eeg_data[512*60:512*60+512 * 60*3,:]

tic = time.perf_counter()
a.get_features_and_reward(eeg_data)
print(len(eeg_data))
toc = time.perf_counter()
print(f"Data handling thread tood: {toc - tic:0.4f} seconds")