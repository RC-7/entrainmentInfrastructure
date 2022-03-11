from eeg_device_interface import EEGDeviceInterface
from aws_messaging_interface import AWSMessagingInterface
from participantInfo import ParticipantInfo
from hdfs5_file_interface import HDFS5FileInterface
from pathlib import Path


class EntrainmentRunner:
    def __init__(self, participant_info):
        self.gtec_device = EEGDeviceInterface()
        self.mi = AWSMessagingInterface()
        self.participant_info = participant_info
        participant_directory = 'data/self.participant_info.participant_ID'
        Path(participant_directory).mkdir(parents=True, exist_ok=True)
        file_path = participant_directory + self.participant_info.participant_ID + f'___{self.participant_info.session}.h5 '
        self.hdfs5_interface = HDFS5FileInterface(file_path)

    def run_entrainment(self):
        # Do the magic
        pass

    def extract_features(self):
        # Do the magic
        pass

    def get_eeg_data(self, dataset_name, time_period, save=True):
        # Do the magic
        pass
