import h5py

from abstract_classes.abstract_file_interface import AbstractFileInterface

EEG_CHANNELS = 64


class HDFS5FileInterface(AbstractFileInterface):

    def __init__(self, filename):
        super().__init__(filename)
        self.hf = None

    def read_file(self):
        hf = h5py.File(self.filename, 'r')
        return hf

    @staticmethod
    def get_specific_data(data, key_path, index_values=None):
        placeholder = data
        for key_value in key_path:
            print(key_value)
            placeholder = placeholder[key_value]
        if index_values is None:
            data = placeholder[()]
        else:
            data = placeholder[index_values]
        return data

    def write_to_file(self, data, options):
        if options is None:
            options = {}
        if self.hf is None:
            self.hf = h5py.File(self.filename, 'a')
        dataset_name = options['dataset_name']
        if dataset_name in self.hf.keys():
            self.hf[dataset_name].resize((self.hf[dataset_name].shape[0] + data.shape[0]), axis=0)
            self.hf[dataset_name][-data.shape[0]:] = data
        else:
            self.hf.create_dataset(dataset_name, data=data, chunks=True, maxshape=(None, EEG_CHANNELS))

        if not options['keep_alive']:
            self.hf.close()
            self.hf = None

    def close_file(self, hdfs_file=None):
        if hdfs_file is None:
            hdfs_file = self.hf
        hdfs_file.close()
