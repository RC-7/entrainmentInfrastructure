from abc import ABCMeta, abstractmethod


class AbstractMlInterface(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def update_entrainment(self, state):
        pass

    @abstractmethod
    def update_model_and_entrainment(self, update_information):
        pass

    @abstractmethod
    def create_model(self, model_name, model_parameters):
        pass

    @abstractmethod
    def save_model(self, model_name):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass


