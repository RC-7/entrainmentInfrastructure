from hdfs5_file_interface import HDFS5FileInterface
import numpy as np
import time

filename = 'test.h5'

data = np.random.rand(512, 64)
print(len(data))

hdfs5_interface = HDFS5FileInterface(filename)

options = {
    'dataset_name': 'raw_data',
    'keep_alive': True
}

write_times = []
time_max = 25 * 60
active_data = []
for i in range(time_max):
    tic = time.perf_counter()
    if len(active_data) == 0:
        active_data = data
    else:
        active_data = np.append(active_data, data, axis=0)
    toc = time.perf_counter()
    print(f"Appending took {toc - tic:0.4f} seconds")
    if len(active_data) > 512:
        options = {
            'dataset_name': 'raw_data',
            'keep_alive': i < time_max - 1
        }
    tic = time.perf_counter()
    hdfs5_interface.write_to_file(data=data, options=options)
    toc = time.perf_counter()
    active_data = []
    elapsed = toc - tic
    write_times.append(elapsed)
    print(f"file write took {elapsed:0.4f} seconds")

print('--------------------------------------------')
print('------------ Summary ---------------')
print(f'Mean: {np.mean(write_times)}')
print(f'Max: {np.max(write_times)}')
print(f'Min: {np.min(write_times)}')
print(f'Std: {np.std(write_times)}')
