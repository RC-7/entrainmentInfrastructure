import sys
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QLabel, QGridLayout, QPushButton, QWidget, QLineEdit
from aws_messaging_interface import AWSMessagingInterface


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

    def delete_elements(self):
        while (child := self.layout.takeAt(0)) is not None:
            child.widget().deleteLater()

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
        [auth_status, response_message] = self.mi.authenticate(request_body_values)
        if auth_status:
            # TODO keep experiment going and prompt participant to make sure they don't have a key
            pass
        else:
            self.active_gui_elements['Warn'].setText(response_message)



    def add_initial_page(self):
        blurb_label = QLabel(self)
        blurb_label.setText('Please enter your details below to sign up or sign in to the experiment.\nEnsure that you '
                            ' enter your secret key if you are returning for the second session.')
        self.layout.addWidget(blurb_label, 0, 1, 1, 2)
        fields = ('name', 'email', 'key')
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
        submit_button.setToolTip('Please ensure that you have submitted your secret key if this is your second session')
        submit_button.clicked.connect(self.submit_authentication_details)
        self.layout.addWidget(submit_button, 4, 1)

        label_warn = QLabel(self)
        label_warn.setText('')
        label_warn.setStyleSheet("QLabel { color : red; }")
        self.active_gui_elements['Warn'] = label_warn
        self.layout.addWidget(label_warn, 5, 1, 2, 2)
