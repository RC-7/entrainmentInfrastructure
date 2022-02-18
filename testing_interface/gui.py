import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel
from aws_messaging_interface import AWSMessagingInterface


class GUI:
    def __init__(self):
        mi = AWSMessagingInterface()
        app = QApplication(sys.argv)
        label = QLabel("Hello World!")
        label.show()
        sys.exit(app.exec_())


def add_initial_page():
    pass