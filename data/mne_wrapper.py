
import numpy as np
import h5py
import mne
import matplotlib as mpl

from matplotlib import pyplot as plt
from util import get_data_csv
from constants import SAMPLING_SPEED, ch_names


def generate_mne_raw_with_info(file_type, file_path, reference=False, scope=''):
    if file_type == 'csv':
        full_eeg_data = get_data_csv(file_path)
        eeg_data = full_eeg_data
    else:
        hf = h5py.File(file_path, 'r')
        if 'hdf5' in file_path:
            tst = hf['RawData']
            tst_samples = tst['Samples']
            eeg_data = tst_samples[()]  # () gets all data
            eeg_data = eeg_data[:, :]
            for i in range(64):
                index = np.index_exp[:, i]
                eeg_data[index] = eeg_data[index] - eeg_data[:, 62]
        else:
            samples = hf['raw_data']
            eeg_data = samples[()]
            ave_ref = (eeg_data[:, 62] + eeg_data[:, 63]) / 2
            if reference:
                for i in range(64):
                    index = np.index_exp[:, i]
                    eeg_data[index] = eeg_data[index] - ave_ref[:]

    ch_types = ['eeg'] * 64

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types,
                           sfreq=SAMPLING_SPEED)
    info.set_montage('standard_1020')  # Will auto set channel names on real cap
    info['description'] = 'My custom dataset'
    raw = mne.io.RawArray(eeg_data.transpose()[0:64], info)
    if scope == 'beta_entrain':
        raw.filter(l_freq=23, h_freq=25)
    if scope == 'beta_entrain_low':
        raw.filter(l_freq=19, h_freq=17)
    if scope == 'alpha_entrain':
        raw.filter(l_freq=10, h_freq=12)
    if scope == 'theta_entrain':
        raw.filter(l_freq=5, h_freq=7)
    if scope == 'beta':
        raw.filter(l_freq=13, h_freq=30)
    if scope == 'alpha':
        raw.filter(l_freq=8, h_freq=13)
    if scope == 'theta':
        raw.filter(l_freq=4, h_freq=8)
    if scope == '':
        raw.filter(l_freq=1., h_freq=50)
    return [raw, info]


def get_events(mne_raw_data):
    return mne.find_events(mne_raw_data)


# def get_fft_mne(data):
#     events = mne.find_events(data, stim_channel='STI 014')


# TODO fix colour map and add reference uV values to it's description
def plot_topo_map(raw_data):
    # Allows for expansion to show time data with map on one figure
    times = np.arange(0.05, 0.151, 0.02)
    fig, ax = plt.subplots(ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)
    # It is possible to set own thresholds here
    [a, b] = mne.viz.plot_topomap(raw_data.get_data()[:, 0], raw_data.info, axes=ax,
                                  show=False, sensors=True, ch_type='eeg')
    cmap = a.cmap
    bounds = b.levels
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.set_title('Topographical map of data', fontweight='bold')
    cax = fig.add_axes([0.85, 0.031, 0.03, 0.8])  # fix location
    # cax = plt.axes([0.85, 0.031, 0.03, 0.8])  # fix location
    plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax,
        boundaries=[-150] + bounds + [120],  # Adding values for extensions.
        extend='neither',
        ticks=bounds,
        # spacing='proportional',
        # orientation='horizontal',
        # label='Discrete intervals, some other units',
    )
    plt.show()


def plot_sensor_locations(raw_data):
    fig, ax = plt.subplots(ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)

    # we plot the channel positions with default sphere - the mne way
    raw_data.plot_sensors(axes=ax, show=False)
    ax.set_title('Channel projection', fontweight='bold')
    plt.show()