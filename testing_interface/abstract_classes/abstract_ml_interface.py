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
    def create_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


