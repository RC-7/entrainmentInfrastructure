import pygds
import json
import numpy as np

from matplotlib import pyplot as plt
from pygds import Scope
from pygds import GDSError
from abstract_classes.abstract_eeg_device_interface import AbstractEEGDeviceInterface

electrodes_file = open('constants/electrodes.json')
DEFAULT_ELECTRODES = json.loads(electrodes_file)['active_electrodes']


def create_active_electrode_bool_array(active_electrode_numbers):
    active_electrodes = []
    for i in range(64):
        active_electrodes.append((i + 1) in active_electrode_numbers)
    return active_electrodes


class EEGDeviceInterface(AbstractEEGDeviceInterface):
    def __init__(self, active_electrodes=DEFAULT_ELECTRODES):
        self.active_electrodes = create_active_electrode_bool_array(active_electrodes)
        self.eeg_device = pygds.GDS()
        self.configure()
        self.print_all_device_info()
        self.active_data = np.zeros([1, sum(self.active_electrodes)])

    # TODO expand for other Gtec devices
    # TODO make dynamic for selecting filters
    def configure(self):
        self.eeg_device.HoldEnabled = 0
        all_sampling_rates = sorted(self.eeg_device.GetSupportedSamplingRates()[0].items())
        f_s_2 = all_sampling_rates[1]
        self.eeg_device.SamplingRate, self.eeg_device.NumberOfScans = f_s_2
        # get all applicable filters
        N = [x for x in self.eeg_device.GetNotchFilters()[0] if x['SamplingRate']
             == self.eeg_device.SamplingRate]
        BP = [x for x in self.eeg_device.GetBandpassFilters()[0] if x['SamplingRate']
              == self.eeg_device.SamplingRate]
        # Set first filter
        for ch in self.eeg_device.Channels:
            ch.Acquire = True
        if N:
            ch.NotchFilterIndex = N[0]['NotchFilterIndex']
        if BP:
            ch.BandpassFilterIndex = BP[0]['BandpassFilterIndex']
        # Needed only for testing
        print('-------------------------------')
        print('sampling rates')
        print(all_sampling_rates)
        print('notch filters')
        print(N)
        print('bandpass filters')
        print(BP)
        print('-------------------------------')
        self.eeg_device.SetConfiguration()

    def impedance_check(self):
        impedance_values = self.eeg_device.GetImpedence(active=self.active_electrodes)
        return impedance_values

    def get_data(self, number_of_cycles=1):
        self.active_data = np.zeros([number_of_cycles, sum(self.active_electrodes)])
        data_received_cycles = 0

        while data_received_cycles < number_of_cycles:
            self.active_data[data_received_cycles] = self.eeg_device.GetData(self.eeg_device.SamplingRate)
            data_received_cycles += 1

    def save_active_data_to_file(self, filename):
        np.save(filename, self.active_data)

    # TODO change first argument of Scope to be dynamic based on range we want to see
    def display_data(self, number_of_cycles, static=False):
        Scope(1 / self.eeg_device.SamplingRate, modal=static)(self.eeg_device.GetData(self.eeg_device.SamplingRate))
        plt.show()

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
