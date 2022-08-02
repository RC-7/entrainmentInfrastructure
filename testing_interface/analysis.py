import numpy as np
import mne
import signal
from collections import defaultdict
from scipy import signal

SAMPLING_SPEED = 512


def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    ma = np.convolve(data, window, 'same')
    inaccurate_window = int(np.ceil(window_size / 2))
    ma = ma[inaccurate_window: -inaccurate_window]
    return ma


class Analysis:

    def __init__(self):
        self.ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'F1', 'AF8', 'F7', 'F5', 'F3', 'AF4', 'Fz', 'F2', 'F4',
                         'F6', 'F8',
                         'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz',
                         'C2', 'C4',
                         'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP4', 'CPz', 'CP2', 'CP1', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                         'P1', 'Pz',
                         'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1',
                         'A2']

        self.tp_indices = [np.index_exp[:, self.ch_names.index('TP7')], np.index_exp[:, self.ch_names.index('TP8')]]

        self.tp_index = [self.ch_names.index('TP7'), self.ch_names.index('TP8')]

    def get_features_and_reward(self, data, entrainment_frequency):
        entrainment_frequency_int = int(entrainment_frequency)
        upper_f_range = entrainment_frequency_int + 1
        lower_f_range = entrainment_frequency_int - 1
        ch_types = ['eeg'] * 2
        ave_ref = (data[:, 62] + data[:, 63]) / 2
        for index in self.tp_indices:
            data[index] = data[index] - ave_ref[:]
        info = mne.create_info(ch_names=['TP7', 'TP8'], ch_types=ch_types,
                               sfreq=SAMPLING_SPEED)
        raw = mne.io.RawArray(data.transpose()[self.tp_index], info)
        raw.filter(l_freq=lower_f_range, h_freq=upper_f_range)

        data = raw.get_data().transpose()

        region_averaged_ma = defaultdict(list)
        for i in range(len(self.tp_indices)):
            index = np.index_exp[:, i]
            electrode_data = data[index]
            samples_per_ft = 100
            overlap = 10
            f, t, Zxx = signal.stft(x=electrode_data, fs=SAMPLING_SPEED, nperseg=samples_per_ft, noverlap=overlap,
                                    nfft=512)
            abs_power = np.abs(Zxx)
            band_averaged_values = []
            for j in range(len(abs_power[0])):
                band_values = []
                for z in range(len(f)):
                    if upper_f_range >= f[z] >= lower_f_range:
                        band_values.append(abs_power[z, j])
                averaged_band_value = np.mean(band_values)
                band_averaged_values.append(averaged_band_value)
            time_averaged = []
            window_averaging = 20
            for j in range(0, len(band_averaged_values), window_averaging):
                if j + window_averaging < len(band_averaged_values):
                    end_index = j + window_averaging
                else:
                    end_index = len(band_averaged_values) - 1
                if j == end_index:
                    break
                values = band_averaged_values[j:end_index]
                time_averaged.append(np.mean(values))
            ma = moving_average(time_averaged, 20)
            region_averaged_ma["T"].append(ma)
        ma_avg = np.mean(region_averaged_ma["T"], axis=0)
        ma_basis = [x - 1.4 for x in ma_avg]
        dy = np.diff(ma_basis)
        increasing_dir_values = len(dy[dy > 0])
        decreasing_dir_values = len(dy[dy < 0])
        total_dir_values = len(dy)
        reward = 0
        fractional_change = abs((ma_avg[-1] - ma_avg[0]) / ma_avg[0])
        if fractional_change > 0.08:
            if increasing_dir_values > decreasing_dir_values:
                reward = increasing_dir_values / total_dir_values
            elif decreasing_dir_values > increasing_dir_values:
                reward = decreasing_dir_values / total_dir_values
            elif increasing_dir_values == decreasing_dir_values:
                reward = 0
        else:
            reward = 0.5
        print(f'reward: {reward}')
        if (ma_avg[-1] - ma_avg[0]) > 0:
            state = 'up'
        else:
            state = 'down'
        print(f'state: {state}')
        return reward, state
