from abc import ABCMeta, abstractmethod


class AbstractMlInterface(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update_entrainment(self, features):
        pass

    @abstractmethod
    def update_model(self, update_information):
        pass

    @abstractmethod
    def create_model(self, update_information):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


