import sys
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QLabel, QGridLayout, QPushButton, QWidget, QLineEdit
from aws_messaging_interface import AWSMessagingInterface
from participantInfo import ParticipantInfo
from info_window import InfoWindow
from util import get_timestamp


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entrainment experiment")
        self.mi = AWSMessagingInterface()
        self.setMinimumSize(QSize(320, 140))
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.active_gui_elements = {}
        self.add_initial_page()
        self.participant_info = ParticipantInfo()
        self.info_window = []

    def delete_elements(self):
        while (child := self.layout.takeAt(0)) is not None:
            child.widget().deleteLater()
        self.active_gui_elements = {}

    def display_get_participant_info(self):
        blurb = 'Please enter your information below. \nThis information is purely used for data analysis and nothing ' \
                'further. \n Experience refers to how many hours you have played a musical instrument before.'
        fields = ('sex', 'age', 'experience')
        self.add_generic_field_entry(blurb, fields, self.submit_participant_data)
        self.resize(550, 200)

    def submit_participant_data(self):
        data = {'sex': self.active_gui_elements['sex'].text(), 'age': self.active_gui_elements['age'].text(),
                'experience': self.active_gui_elements['experience'].text()}
        missing_fields = []
        for key in data:
            if data[key] == '':
                missing_fields.append(key)
        int_fields = ['age', 'experience']
        for field in int_fields:
            if field not in missing_fields:
                try:
                    data[field] = int(data[field])
                except ValueError:
                    missing_fields.append(field)
        if len(missing_fields) != 0:
            warn_message = 'Please enter a valid value for the following: ' + ','.join(missing_fields)
            self.active_gui_elements['Warn'].setText(warn_message)
            self.resize(550, 200)
            return
        data['participantID'] = self.participant_info.participant_ID
        data['timestamp'] = get_timestamp()
        data_type = 'PData'
        [status, _] = self.mi.send_data(data_type, data)
        if status:
            self.participant_info.info = data
            self.delete_elements()
            self.display_start_experiment()

    def display_start_experiment(self):
        print('lets get going')

    def submit_authentication_details(self):
        name = self.active_gui_elements['name'].text()
        email = self.active_gui_elements['email'].text()
        key = self.active_gui_elements['key'].text()

        if name == '':
            self.active_gui_elements['Warn'].setText('Please enter your Name to continue')
            return
        if email == '' and key == '' and name != '':
            self.active_gui_elements['Warn'].setText('Please enter your email if it is your first session and your '
                                                     'secret key if it is your second session')
            self.resize(550, 200)
            return

        # TODO Add additional check to ensure that the participant is first time if no key is entered
        request_body_values = {
            'name': name,
            'email': email,
            'secret_key': key
        }
        self.active_gui_elements['Warn'].setText('')
        request_body_values = dict([(k, v) for k, v in request_body_values.items() if v != ''])
        # UNCOMMENT ME!!
        # [auth_status, response_message] = self.mi.authenticate(request_body_values)
        auth_status = True
        response_message = {"participant_ID": "yes", "message": "Participant ID created, check your email for your secret "
                                                             "code", "group": "C", "session": 1}

        if auth_status:
            self.info_window = InfoWindow(response_message['message'])
            self.info_window.setStyleSheet("QLabel { color : green; }")
            self.info_window.show()
            # self.active_gui_elements['Warn'].setStyleSheet("QLabel { color : green; }")
            # self.active_gui_elements['Warn'].setText(response_message['message'])
            # time.sleep(3)
            self.participant_info.set_from_auth_response(response_message)
            self.delete_elements()
            if self.participant_info.session == 1:
                self.display_get_participant_info()
            else:
                self.display_start_experiment()
        else:
            self.active_gui_elements['Warn'].setText(response_message)

    def add_initial_page(self):
        blurb = 'Please enter your details below to sign up or sign in to the experiment.\nEnsure that you  enter ' \
                'your secret key if you are returning for the second session. '
        fields = ('name', 'email', 'key')
        self.add_generic_field_entry(blurb, fields, self.submit_authentication_details)

    def add_generic_field_entry(self, blurb, fields, onsubmit, tooltip=None):
        blurb_label = QLabel(self)
        blurb_label.setText(blurb)
        self.layout.addWidget(blurb_label, 0, 1, 1, 2)
        counter = 1
        for field in fields:
            label = QLabel(self)
            label_value = field.capitalize()
            label.setText(label_value + ':')
            self.layout.addWidget(label, counter, 1)
            line_name = QLineEdit(self)
            line_name.resize(200, 32)
            self.layout.addWidget(line_name, counter, 2)
            self.active_gui_elements[field] = line_name
            counter += 1

        submit_button = QPushButton("Submit")
        if tooltip is not None:
            submit_button.setToolTip(tooltip)
        submit_button.clicked.connect(onsubmit)
        self.layout.addWidget(submit_button, counter + 1, 1)

        label_warn = QLabel(self)
        label_warn.setText('')
        label_warn.setStyleSheet("QLabel { color : red; }")
        self.active_gui_elements['Warn'] = label_warn
        self.layout.addWidget(label_warn, counter + 2, 1, 2, 2)
