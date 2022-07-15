import pandas as pd
import numpy as np
import random
import os

from abstract_classes.abstract_ml_interface import AbstractMlInterface


class QLearningInterface(AbstractMlInterface):
    def __init__(self, model_path = '.', model_parameters = None, model_name = None):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.model_parameters = model_parameters
        if model_path and not model_parameters:
            self.load_model()
        elif model_parameters and model_path:
            self.create_model(model_path, model_parameters)
        else:
            raise Exception('Not enough parameters passed to Q learning \
                interface to initialise model')

    def update_entrainment(self, features):
        pass

    def update_model(self, update_information):
        pass

    def read_parameters(self, model_parameters):
        self.states = model_parameters['states']
        self.actions = model_parameters['actions']
        self.epsilon = model_parameters['epsilon']
        self.learning_rate = model_parameters['learning_rate']
        self.discount_factor = model_parameters['discount_factor']

    # Pass model perameters as a dict of state and action arrays
    def create_model(self, model_path, model_parameters):
        self.read_parameters(model_parameters)

        state_zeros = np.zeros(len(self.states))
        state_zeros[0] = 1
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
            action = random.choice(self.actions)
            return action
        else:
            action = self.model.loc[state].idxmax(axis=1)
            return action



    def update_q_value(self):
        pass


    def save_model(self):
        old_model_name = self.model_name.split("_")
        if len(old_model_name) == 2:
            new_model_name = f'{old_model_name[0]}_{int(old_model_name[1]) + 1}'
        else:
            new_model_name = f'{old_model_name[0]}_0'

        self.model_name = new_model_name
        path = self.model_path + self.model_name
        self.model.to_csv(path, index=True)

    def load_model(self):
        path = f'{self.model_path}'
        f = lambda s : self.model_name in s
        filenames = list(filter(f, os.listdir(self.model_path)))
        self.model_name = filenames[0]
        self.model = pd.read_csv(self.model_path + self.model_name, index_col=0)
        print(self.model)


