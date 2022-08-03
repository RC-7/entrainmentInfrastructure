from Q_learning_interface import QLearningInterface
import h5py
import pandas as pd
from math import floor

model_parameters = {
    "states":  ['up_24', 'down_24', 'up_18', 'down_18'],
    'actions': ['24', '18'],
    "epsilon": 1,
    "learning_rate": 0.2,
    "discount_factor": 0.4,
    "step": 0
}

q_learn = QLearningInterface(model_parameters=model_parameters, model_path='models/', model_name='bciAgent')
training_ds_names = ['Full_run_S', 'Full_run_A', 'Full_run_J', 'Full_run_D']
ds_name = 'ml_beta_audio'
p_count = 1
increase_rewarding = False
save_intermittent_models = True
for p in training_ds_names:
    count_lookup = [5, 3, 1, 4]
    count = count_lookup[p_count - 1]
    filename = f'../data/custom_suite/{p}/{ds_name}.h5'
    actions = f'../data/custom_suite/{p}/bciAgent_{count}_actions.csv'
    p_count += 1
    q_learn.current_entrainment = '24'
    q_learn.current_index = ''
    previous_actions = pd.read_csv(actions)
    action = '24'
    for i in range(5):
        hf = h5py.File(filename, 'r')
        samples = hf['raw_data']
        eeg_data = samples[()]
        hf.close()
        if not increase_rewarding:
            eeg_data = eeg_data[512*60*3*i:512*60*3*(i+1), :]
            q_learn.update_model_and_entrainment(eeg_data)
            action = previous_actions.loc[i]['action']
            q_learn.current_entrainment = f'{action}'
        else:
            eeg_data = eeg_data[512*60*3*i:floor(512*60*3*(i+0.5)), :]
            current_action = q_learn.current_entrainment
            q_learn.update_model_and_entrainment(eeg_data)
            if i == 0:
                action = '24'
                q_learn.current_entrainment = f'{action}'
            else:
                q_learn.current_entrainment = f'{current_action}'
            hf = h5py.File(filename, 'r')
            samples = hf['raw_data']
            eeg_data = samples[()]
            hf.close()
            eeg_data = eeg_data[floor(512*60*3*(i+0.5)):512*60*3*(i+1), :]
            q_learn.update_model_and_entrainment(eeg_data)
            action = previous_actions.loc[i]['action']
            q_learn.current_entrainment = f'{action}'
    if save_intermittent_models:
        q_learn.save_model()
if not save_intermittent_models:
    q_learn.save_model()
