from operator import mod
from Q_learning_interface import QLearningInterface
import h5py
import time
from participant_info import test_Q_datapoint

model_parameters = {
    "states":  ['up_24', 'down_24', 'up_18', 'down_18'],
    'actions': ['24', '18'],
    "epsilon": 1,
    "learning_rate": 0.2,
    "discount_factor": 0.4,
    "step": 0
}
q_learn = QLearningInterface(model_parameters=model_parameters, model_path='models/', model_name='bciAgent_0')
# q_lean.save_model()


hf = h5py.File(f'../data/custom_suite/{test_Q_datapoint}/beta_audio.h5', 'r')

samples = hf['raw_data']
eeg_data = samples[()]
eeg_data = eeg_data[512*60:512*60+512*60*3,:]


tic = time.perf_counter()
q_learn.update_model_and_entrainment(eeg_data)
toc = time.perf_counter()
print(f"Data handling thread took: {toc - tic:0.4f} seconds")
print(q_learn.model)
samples = hf['raw_data']
eeg_data = samples[()]
eeg_data = eeg_data[512*60*3:512*60*6,:]


tic = time.perf_counter()
q_learn.update_model_and_entrainment(eeg_data)
toc = time.perf_counter()
print(f"Data handling thread tood: {toc - tic:0.4f} seconds")
print(q_learn.model)

samples = hf['raw_data']
eeg_data = samples[()]
eeg_data = eeg_data[512*60*6:512*60+512*60*9,:]

tic = time.perf_counter()
q_learn.update_model_and_entrainment(eeg_data)
toc = time.perf_counter()
print(f"Data handling thread tood: {toc - tic:0.4f} seconds")
# q_learn.save_model()
#
# print(q_learn.model)
# samples = hf['raw_data']
# eeg_data = samples[()]
# eeg_data = eeg_data[512*60*9:512*60+512*60*12,:]
#
# tic = time.perf_counter()
# q_learn.update_model_and_entrainment(eeg_data)
# toc = time.perf_counter()
# print(f"Data handling thread tood: {toc - tic:0.4f} seconds")
#
# print(q_learn.model)
# print(q_learn.current_entrainment)
# print(q_learn.current_index)

# q_learn.update_entrainment('up')
# q_learn.bellmans('up_24', 0.2)

# q_learn.save_model()