from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class InfoWindow(QWidget):
    def __init__(self, info):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel(info)
        layout.addWidget(self.label)
        self.setLayout(layout)
