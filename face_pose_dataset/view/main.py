from PySide2 import QtWidgets

from face_pose_dataset import view


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


class MainWidget(QtWidgets.QWidget):
    def __init__(self, storage):
        QtWidgets.QWidget.__init__(self)

        # Global data store

        self.video = view.VideoWidget(360, 240)
        self.plot = view.MatplotlibWidget(storage)
        self.controls = ControlWidget()

        self.layout = QtWidgets.QGridLayout()

        self.layout.addWidget(self.plot, 0, 0, 2, 4)
        self.layout.addWidget(self.video, 0, 4, 1, 2)
        self.layout.addWidget(self.controls, 1, 4, 1, 2)

        self.setLayout(self.layout)
