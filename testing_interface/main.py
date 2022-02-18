from gui import GUI
from PySide6.QtWidgets import QApplication, QLabel, QHBoxLayout, QPushButton,  QWidget
import sys


def main():
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
