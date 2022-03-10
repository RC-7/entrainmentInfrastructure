from abc import ABCMeta, abstractmethod


class AbstractFileInterface(metaclass=ABCMeta):

    def __init__(self, filename):
        self.filename = filename

    @abstractmethod
    def read_file(self, number_of_cycles):
        pass

    @abstractmethod
    def write_to_file(self, data, options={}):
        pass

    def set_filename(self, filename):
        self.filename = filename



