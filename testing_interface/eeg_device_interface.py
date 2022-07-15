import datetime
import pygds
import json
import numpy as np
import time
from matplotlib import pyplot as plt
from pygds import Scope
from pygds import GDSError
from abstract_classes.abstract_eeg_device_interface import AbstractEEGDeviceInterface
from csv_file_interface import CSVFileInterface
from hdfs5_file_interface import HDFS5FileInterface
from Q_learning_interface import QLearningInterface

electrodes_file = open('constants/electrodes.json')
DEFAULT_ELECTRODES = json.load(electrodes_file)['active_electrodes']

SAMPLING_RATE = 512
# Twenty seconds of samples saved at a time
INTERMEDIATE_SAMPLE_WRITE_THRESHOLD = 512*10
INTERMEDIATE_SECOND_WRITE_THRESHOLD = 20

ML_CONSIDERATION_THRESHOLD = SAMPLING_RATE * 60 * 3


def create_active_electrode_bool_array(active_electrode_numbers):
    active_electrodes = []
    for i in range(64):
        active_electrodes.append((i + 1) in active_electrode_numbers)
    return active_electrodes


class EEGDeviceInterface(AbstractEEGDeviceInterface):
    def __init__(self, active_electrodes=DEFAULT_ELECTRODES, testing=False):
        self.active_electrodes = create_active_electrode_bool_array(active_electrodes)
        self.eeg_device = pygds.GDS()
        self.configure(testing)
        self.filter_values = 0
        model_parameters = {
            "states":  ['up_24', 'down_24', 'up_27', 'down_27'], 
            'actions': ['24', '27'],
            "epsilon": 1,
            "learning_rate": 0.2,
            "discount_factor": 0.4,
            "step": 0
        }

        self.q_learn_agent = QLearningInterface(model_parameters=model_parameters,
                                                model_path='models/', model_name='bciAgent')
        # self.print_all_device_info()
        self.active_data = []
        self.current_ml_data = []
        self.data_received_cycles = 0
        self.save_intermediate_data = False
        self.filename = ''
        self.hdfs5_interface = HDFS5FileInterface(self.filename)

    def configure(self, testing):
        # From reference doc
        self.eeg_device.HoldEnabled = 0
        all_sampling_rates = sorted(self.eeg_device.GetSupportedSamplingRates()[0].items())
        f_s_2 = all_sampling_rates[1]
        self.eeg_device.SamplingRate, self.eeg_device.NumberOfScans = f_s_2
        # get all applicable filters for sampling frequency
        notch_filters = [x for x in self.eeg_device.GetNotchFilters()[0] if x['SamplingRate']
                         == self.eeg_device.SamplingRate]
        bandpass_filters = [x for x in self.eeg_device.GetBandpassFilters()[0] if x['SamplingRate']
                            == self.eeg_device.SamplingRate]
        print('---------------------------------------')
        # print('Printing filter information for sampling rate')
        # print(notch_filters)
        # print(bandpass_filters)
        print('Printing sampling rate')
        print(all_sampling_rates)
        print('Printing sampling value')
        print(self.eeg_device.GetSupportedSamplingRates())
        # Test me
        # self.eeg_device.NumberOfScans_calc()
        print(self.eeg_device.DeviceType)
        print('---------------------------------------')
        channel_counter = 1
        if testing:
            self.eeg_device.InternalSignalGenerator.Enabled = True
            self.eeg_device.InternalSignalGenerator.Frequency = 10  # 10 Hz signal for testing purposes
        for ch in self.eeg_device.Channels:
            if channel_counter <= 64:
                ch.Acquire = True
                ch.BipolarChannel = 0
                ch.NotchFilterIndex = notch_filters[0]['NotchFilterIndex']
                # 'Order': 8 'LowerCutoffFrequency': 0.01, 'UpperCutoffFrequency': 100.0
                ch.BandpassFilterIndex = bandpass_filters[14]['BandpassFilterIndex']
                channel_counter += 1
            else:
                ch.Acquire = False

        print(self.eeg_device.Channels())
        self.eeg_device.SetConfiguration()

    def impedance_check(self):
        impedance_values = self.eeg_device.GetImpedance()
        return impedance_values

    def more(self, samples):
        tic = time.perf_counter()
        # Do not save or consider filter start up
        if self.filter_values < 40 * SAMPLING_RATE:
            self.filter_values += len(samples)
            return self.data_received_cycles > 0
        if len(self.active_data) == 0:
            self.active_data = samples
        else:
            self.active_data = np.append(self.active_data, samples, axis=0)
        if len(self.current_ml_data) == 0:
            self.current_ml_data = samples
        else:
            self.current_ml_data = np.append(self.current_ml_data, samples, axis=0)
        if len(self.current_ml_data) >= ML_CONSIDERATION_THRESHOLD:
            # Try to apply ML syncronously
            self.q_learn_agent(self.current_ml_data)
            self.current_ml_data = []
        # Every Two minutes write data in file to keep active memory low
        if self.save_intermediate_data and len(self.active_data) > INTERMEDIATE_SAMPLE_WRITE_THRESHOLD:
            options = {
                'dataset_name': 'raw_data',
                'keep_alive': True
            }
            print(len(self.active_data))
            print(len(self.active_data[0]))
            self.save_active_data_to_file(self.filename, options=options)
            self.active_data = []
            self.data_received_cycles -= INTERMEDIATE_SECOND_WRITE_THRESHOLD
        toc = time.perf_counter()
        print(f"Data handling thread tood: {toc - tic:0.4f} seconds")
        print(self.data_received_cycles)
        return self.data_received_cycles > 0

    def get_scaling(self):
        scaling = self.eeg_device.GetScaling()
        print(scaling)

    def set_scaling(self):
        pass

    def get_data(self, number_of_minutes=1, save_intermediate_data=False, filename=''):
        self.active_data = []
        self.data_received_cycles = number_of_minutes * 60  # Minutes to number of sampling periods
        # Testing increasing first argument, will get half of a minute's worth of samples
        self.save_intermediate_data = save_intermediate_data
        self.filename = filename
        self.eeg_device.GetData(self.eeg_device.SamplingRate, self.more)
        options = {
            'dataset_name': 'raw_data',
            'keep_alive': False
        }
        # self.save_active_data_to_file(self.filename, options=options)
        self.hdfs5_interface.close_file()
        self.active_data = []
        self.save_intermediate_data = False
        self.filename = ''

    def save_data_csv(self, filename, data=None):
        if data is None:
            data = self.active_data
        csv = CSVFileInterface(filename=filename)
        csv.write_to_file(data)

    def save_data_hdfs(self, filename, options, data=None):
        if data is None:
            data = self.active_data
        if self.hdfs5_interface.filename != filename:
            self.hdfs5_interface.set_filename(filename)
        self.hdfs5_interface.write_to_file(data=data, options=options)

    def save_active_data_to_file(self, filename, options=None, data=None):
        if 'h5' in filename:
            self.save_data_hdfs(filename, options, data)
        elif 'csv' in filename:
            self.save_data_csv(filename, data)

    # TODO change first argument of Scope to be dynamic based on range we want to see
    def display_data(self, number_of_cycles, static=False):
        Scope(number_of_cycles / self.eeg_device.SamplingRate, modal=static)(
            self.eeg_device.GetData(self.eeg_device.SamplingRate))
        plt.show()

    def save_config_to_dataset(self, filename):
        config_data = []
        for c in self.eeg_device.Configs:
            config_data.append(str(c))

        config_to_write = {
            'values': config_data,
            'timestamp': str(datetime.datetime.now(datetime.timezone.utc))}

        options = {
            'dataset_name': 'config',
            'keep_alive': False
        }
        self.save_active_data_to_file(filename=filename, options=options, data=config_to_write)

    def save_impedance_to_dataset(self, filename):
        impedance_data = {'values': self.impedance_check(),
                          'timestamp': str(datetime.datetime.now(datetime.timezone.utc))}
        options = {
            'dataset_name': 'impedance',
            'keep_alive': False
        }
        self.save_active_data_to_file(filename=filename, options=options, data=impedance_data)

    def print_all_device_info(self):
        print("Testing communication with the devices")
        print("======================================")
        print()
        self.eeg_device.SetConfiguration()
        # print all Configs
        print("Devices:")
        for c in self.eeg_device.Configs:
            print(str(c))
        print()
        print()
        # calc number of channels
        print("Configured number of channels: ", self.eeg_device.N_ch_calc())
        print()
        # available channels
        print("Available Channels: ", self.eeg_device.GetAvailableChannels())
        print()
        # device info string
        print("Device information:")
        dis = self.eeg_device.GetDeviceInformation()
        for di in dis:
            print(di)
            print()
            print()
            # supported sampling rates
            print("Supported sampling rates: ")
            for sr in self.eeg_device.GetSupportedSamplingRates():
                for x in sr:
                    print(str(x))
            print()
            # impedance values
            print("Measure impedance values: ")
            try:
                imps = self.eeg_device.GetImpedance()
                print(imps)
            except GDSError as e:
                print(e)
            print()
            # filters
            print("Bandpass filters:")
            bps = self.eeg_device.GetBandpassFilters()
            for bp in bps:
                for abp in bp:
                    print(str(abp))
            print()
            print("Notch filters:")
            notch_filters = self.eeg_device.GetNotchFilters()
            for notch in notch_filters:
                for a_notch in notch:
                    print(str(a_notch))

    def close_device(self):
        self.eeg_device.Close()
        self.eeg_device = []

    def set_filtering(self):
        pass
