import pandas as pd
import numpy as np
from random import random, choice
import os
import json
import datetime

from abstract_classes.abstract_ml_interface import AbstractMlInterface
from aws_messaging_interface import AWSMessagingInterface


class QLearningInterface(AbstractMlInterface):
    def __init__(self, participantID='1', model_path = '.', model_parameters = None, model_name = None):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.participantID = participantID
        self.base_entrainment_f = 370
        self.current_entrainment = 24
        #  TODO check 27 Hz
        self.entrainmentLookup = {
            24: 21,
            27: 24
        }
        # self.mi = AWSMessagingInterface()
        if model_path and not model_parameters:
            self.load_model()
        elif model_parameters and model_path:
            self.model_parameters = model_parameters
            self.create_model()
        else:
            raise Exception('Not enough parameters passed to Q learning \
                interface to initialise model')

    def update_entrainment(self, state):
        state_index = state + str(self.current_entrainment)
        action = self.policy_function(state)
        self.current_entrainment = action
        date_type = 'EntrainmentSettings'
        data = {
        'participantID': self.participantID,
        'customEntrainment': {
            "visual": {
                'colour': 'red',
                'frequency': '0',
                },
            'audio': {
                'baseFrequency': self.base_entrainment_f,
                'entrainmentFrequency': self.entrainmentLookup[action],
                },
            'neurofeedback': {
                'redChannel': '0',
                'greenChannel': '0'
                },
            },
        'timestamp': str(datetime.datetime.now(datetime.timezone.utc)),
        'session': '2'
        }
        # TODO save actions and results
        print(action)
        # Commented for testing 
        # self.mi.send_data(date_type, data)

    def update_model(self, update_information):
        # Update with values
        pass

    def update_q_value(self):
        pass

    def read_parameters(self):
        self.states = self.model_parameters['states']
        self.actions = self.model_parameters['actions']
        self.epsilon = self.model_parameters['epsilon']
        self.learning_rate = self.model_parameters['learning_rate']
        self.discount_factor = self.model_parameters['discount_factor']

    def get_parameters(self):
        model_parameters = {}
        model_parameters['states'] = self.states
        model_parameters['actions'] = self.actions
        model_parameters['epsilon'] = self.epsilon
        model_parameters['learning_rate'] = self.learning_rate
        model_parameters['discount_factor'] = self.discount_factor
        return model_parameters

    # Pass model perameters as a dict of state and action arrays
    def create_model(self):
        self.read_parameters()

        state_zeros = np.zeros(len(self.states))
        data = {}
        for action in self.actions:
            data[action] = state_zeros
        
        self.model = pd.DataFrame(data, index=self.states)
        print(self.model)
        print(self.model.loc['up_24'][24])
        print(self.model.loc['up_24'].idxmax(axis=1))

    def policy_function(self, state):
        rand_value = random()
        
        if rand_value < self.epsilon:
            # TODO decide if want to ignore already explored actions
            action = choice(self.actions)
            return action
        else:
            action = self.model.loc[state].idxmax(axis=1)
            return action


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



    def load_model(self):
        path = f'{self.model_path}'
        f = lambda s : len(s.split("_")) == 2 and self.model_name in s
        filenames = list(filter(f, os.listdir(self.model_path)))
        self.model_name = filenames[0]
        full_model_path = self.model_path + self.model_name
        self.model = pd.read_csv(self.model_path + self.model_name, index_col=0)
        parameter_path = full_model_path + '_Parameters.json'
        parameter_file = open(parameter_path)
        self.model_parameters = json.load(parameter_file)
        parameter_file.close()
        self.read_parameters()


