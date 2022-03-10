
from hdfs5_file_interface import HDFS5FileInterface
import numpy as np


hdfs5_file = HDFS5FileInterface('test.h5')

data = np.random.random(size = (100,33))

options = {
    'dataset_name' : 'extra_data',
    'keep_alive': False
}

hdfs5_file.write_to_file(data=data,options=options)

data_values = hdfs5_file.read_file()
print(data_values)
data_raw = hdfs5_file.get_specific_data(data=data_values, key_path=['extra_data'], index_values=None)

print('--------------- After first add ---------------')
print(len(data_raw))
print(len(data_raw[0]))

hdfs5_file.close_file(data_values)

options = {
    'dataset_name' : 'raw_data',
    'keep_alive': False
}

data = np.random.random(size = (10,33))

hdfs5_file.write_to_file(data=data,options=options)

options = {
    'dataset_name' : 'extra_data',
    'keep_alive': False
}

data = np.random.random(size = (100,33))

hdfs5_file.write_to_file(data=data,options=options)

data_values = hdfs5_file.read_file()
print(data_values)
data_raw = hdfs5_file.get_specific_data(data=data_values, key_path=['raw_data'], index_values=None)


print('--------------- After Second add add ---------------')
print('--------------- raw data ---------------')
print(len(data_raw))
print(len(data_raw[0]))

data_raw = hdfs5_file.get_specific_data(data=data_values, key_path=['extra_data'], index_values=None)

print('--------------- extra data ---------------')
print(len(data_raw))
print(len(data_raw[0]))