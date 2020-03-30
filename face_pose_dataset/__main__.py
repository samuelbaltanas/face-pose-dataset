import sys

from PySide2 import QtWidgets

from face_pose_dataset import pose_storage, ui


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setWindowTitle("Tutorial")
        # self.setGeometry(300, 200, 500, 400)


class MainWidget(QtWidgets.QWidget):
    def __init__(self, data):
        QtWidgets.QWidget.__init__(self)

        # Global data store
        self._data = data

        self.video = ui.VideoWidget(360, 240)
        self.plot = ui.MatplotlibWidget(storage)
        self.controls = ui.ControlWidget()

        self.layout = QtWidgets.QGridLayout()

        self.layout.addWidget(self.plot, 0, 0, 2, 2)
        self.layout.addWidget(self.video, 0, 2, 1, 2)
        self.layout.addWidget(self.controls, 1, 2, 1, 2)

        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    dims = 7, 7
    yaw_range = pose_storage.DEFAULT_YAW_RANGE
    pitch_range = pose_storage.DEFAULT_PITCH_RANGE

    storage = pose_storage.ScoreMatrix(
        dims, pitch_range=pitch_range, yaw_range=yaw_range
    )

    window = MainWindow()
    window.resize(800, 600)

    # Central widget
    widget = MainWidget(storage)
    window.setCentralWidget(widget)

    window.show()

    # Execute application
    sys.exit(app.exec_())
