from util import get_hemisphere

ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'F1', 'AF8', 'F7', 'F5', 'F3', 'AF4', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP4', 'CPz', 'CP2', 'CP1', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
            'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'F9', 'F10', 'A1', 'A2']

eeg_bands = {'Delta': (0.5, 4),
             'Theta': (4, 8),
             'Alpha': (8, 13),
             'Beta': (13, 30),
             'Gamma': (30, 45)}

SAMPLING_SPEED = 512

ch_hemisphere = get_hemisphere(ch_names)

power_analysis_file = 'power_summary.csv'
