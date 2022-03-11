

class ParticipantInfo:
    def __init__(self):
        self.participant_ID = ''
        self.session = ''
        self.group = ''
        self.info = {}

    def set_from_auth_response(self, response):
        self.participant_ID = response['participant_ID']
        self.session = response['session']
        self.group = str(response['group'])
