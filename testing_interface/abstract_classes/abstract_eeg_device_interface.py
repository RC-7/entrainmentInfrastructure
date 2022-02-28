from abc import ABCMeta, abstractmethod


class AbstractEEGDeviceInterface(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_data(self, number_of_cycles):
        pass

    @abstractmethod
    def impedance_check(self):
        pass

    @abstractmethod
    def set_filtering(self):
        pass

    @abstractmethod
    def save_active_data_to_file(self, filename):
        pass

    @abstractmethod
    def close_device(self):
        pass

    @abstractmethod
    def display_data(self, number_of_cycles, static):
        pass




