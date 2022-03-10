from numpy import genfromtxt
from numpy import save

from abstract_classes.abstract_file_interface import AbstractFileInterface


class CSVFileInterface(AbstractFileInterface):

    def read_file(self):
        data = genfromtxt(self.filename, delimiter=',')
        return data

    def write_to_file(self, data, options=None):
        save(self.filename, data)
