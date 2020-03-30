from PySide2 import QtWidgets


class ControlWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.buttons = [
            QtWidgets.QPushButton("Pause / Resume"),
            QtWidgets.QPushButton("Finish"),
            # QtWidgets.QPushButton(),
        ]

        self.layout = QtWidgets.QGridLayout()

        for i, wid in enumerate(self.buttons):
            self.layout.addWidget(wid, i, 0)

        self.setLayout(self.layout)
