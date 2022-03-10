from abc import ABCMeta, abstractmethod


class AbstractFileInterface(metaclass=ABCMeta):

    def __init__(self, filename):
        self.filename = filename

    @abstractmethod
    def read_file(self):
        pass

    @abstractmethod
    def write_to_file(self, data, options):
        if options is None:
            options = {}

    def set_filename(self, filename):
        self.filename = filename



