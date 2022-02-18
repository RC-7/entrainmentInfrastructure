import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel


class Interface:
    def __init__(self):
        app = QApplication(sys.argv)

        sys.exit(app.exec_())

def add_initial_page():
    pass