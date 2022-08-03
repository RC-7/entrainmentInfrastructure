import pandas as pd
import numpy as np
from random import random, choice
import os
import json
import datetime

from abstract_classes.abstract_ml_interface import AbstractMlInterface
from aws_messaging_interface import AWSMessagingInterface
from analysis import Analysis


class QLearningInterface(AbstractMlInterface):
    def __init__(self, participantID='1', model_path='.', model_parameters=None, model_name=None):
        super().__init__()
        self.states = []
        self.actions = []
        self.epsilon = 1
        self.learning_rate = 1
        self.discount_factor = 1
        self.step = 0
        # Decaying epsilon constants
        self.n_epsilon = 20
        self.pend = 0.15
        self.pinit = 1
        self.model_path = model_path
        self.model_name = model_name
        self.participantID = participantID
        self.base_entrainment_f = 370
        # Always start with a 24 Hz BB
        self.current_entrainment = '24'
        self.actions_taken = pd.DataFrame()
        self.current_index = ''
        self.analyser = Analysis()
        self.entrainmentLookup = {
            '24': 21,
            '18': 20
        }
        self.mi = AWSMessagingInterface()
        if model_path and not model_parameters:
            self.load_model()
        elif model_parameters and model_path:
            self.model_parameters = model_parameters
            self.create_model()
        else:
            raise Exception('Not enough parameters passed to Q learning \
                interface to initialise model')

    def set_entrainment_DB_item(self, action):
        print(f'updating entrainment to: {action}')
        data_type = 'EntrainmentSettings'
        data = {
            'participantID': str(self.participantID),
            'customEntrainment': {
                "visual": {
                    'colour': 'red',
                    'frequency': '0',
                },
                'audio': {
                    'baseFrequency': str(self.base_entrainment_f),
                    'entrainmentFrequency': str(self.entrainmentLookup[action]),
                },
                'neurofeedback': {
                    'redChannel': '0',
                    'greenChannel': '0'
                },
            },
            'timestamp': str(datetime.datetime.now(datetime.timezone.utc)),
            'session': '2'
        }
        self.mi.send_data(data_type, data)

    def update_entrainment(self, state):
        # Has last action and persists it
        state_index = state + '_' + str(self.current_entrainment)
        print(f'State index: {state_index}')
        action = self.policy_function(state_index)
        print(f'Next action: {action}')
        self.current_entrainment = action
        self.current_index = state_index
        self.set_entrainment_DB_item(action)
        epoched_values = {
            'state': state,
            'action': action,
            'timestamp': str(datetime.datetime.now(datetime.timezone.utc))
        }
        self.actions_taken = self.actions_taken.append(epoched_values, ignore_index=True)
        print(action)

    # Pass in all EEG data to get new Q values and iterate entrainment
    def update_model_and_entrainment(self, data):
        print('Updating Machine learning model')
        reward, state = self.analyser.get_features_and_reward(data, self.current_entrainment)
        if self.current_index != '':
            new_state = state + '_' + self.current_entrainment
            self.bellmans(new_state, reward)
            print('--------------------')
            print('Updated Q Table')
            print(self.model)
            print('--------------------')
        self.update_entrainment(state)

    def bellmans(self, new_state, reward):
        Qs = self.model.loc[self.current_index][self.current_entrainment]
        Qs1 = max(self.model.loc[new_state].values)
        Qnew = Qs + self.learning_rate * (reward + self.discount_factor * Qs1 - Qs)
        self.model.at[self.current_index, self.current_entrainment] = Qnew

    def read_parameters(self):
        self.states = self.model_parameters['states']
        self.actions = self.model_parameters['actions']
        self.epsilon = self.model_parameters['epsilon']
        self.learning_rate = self.model_parameters['learning_rate']
        self.discount_factor = self.model_parameters['discount_factor']
        self.step = self.model_parameters['step']

    def get_parameters(self):
        model_parameters = {'states': self.states, 'actions': self.actions, 'epsilon': self.epsilon,
                            'learning_rate': self.learning_rate, 'discount_factor': self.discount_factor,
                            'step': self.step}
        return model_parameters

    def create_model(self):
        print('creating new model!!')
        self.read_parameters()

        state_zeros = np.zeros(len(self.states))
        data = {}
        for action in self.actions:
            data[action] = state_zeros

        self.model = pd.DataFrame(data, index=self.states)
        print(self.model)

    def policy_function(self, state):
        rand_value = random()
        if self.epsilon != 0 and self.epsilon != self.pend:
            self.update_epsilon()
        if rand_value <= self.epsilon:
            action = str(choice(self.actions))
            return action
        else:
            action = str(self.model.loc[state].idxmax(axis=0))
            return action

    def update_epsilon(self):
        r = max((self.n_epsilon - self.step) / self.n_epsilon, 0)
        self.epsilon = (self.pinit - self.pend) * r + self.pend
        self.step += 1

    def save_model(self):
        old_model_name = self.model_name.split("_")
        if len(old_model_name) == 2:
            new_model_name = f'{old_model_name[0]}_{int(old_model_name[1]) + 1}'
        else:
            new_model_name = f'{old_model_name[0]}_0'

        self.model_name = new_model_name
        path_model = self.model_path + self.model_name
        self.model.to_csv(path_model, index=True)
        path_parameters = path_model + '_Parameters.json'
        parameters = self.get_parameters()
        with open(path_parameters, "w") as outfile:
            json.dump(parameters, outfile)
        self.save_actions(path_model)

    def save_actions(self, path_model):
        actions_file = path_model + '_actions.csv'
        self.actions_taken.to_csv(actions_file, index=False)

    def load_model(self):
        print('Using existing model!!')
        f = lambda s: len(s.split("_")) == 2 and self.model_name in s
        filenames = list(filter(f, os.listdir(self.model_path)))
        self.model_name = filenames[0]
        # print(filenames)
        full_model_path = self.model_path + self.model_name
        self.model = pd.read_csv(self.model_path + self.model_name, index_col=0)
        parameter_path = full_model_path + '_Parameters.json'
        parameter_file = open(parameter_path)
        self.model_parameters = json.load(parameter_file)
        print(self.model)
        parameter_file.close()
        self.read_parameters()
