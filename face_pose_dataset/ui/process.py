import time

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from face_pose_dataset import camera

class CameraThread(QtCore.QThread):
    change_pixmap = QtCore.Signal(np.ndarray)

    def __init__(self, width=640, height=480, delay=1 / 20):
        super().__init__()
        self.width = width
        self.height = height
        self.delay = delay

    def run(self):
        with camera.AstraCamera() as cam:
            while True:
                frame = cam.read_rgb()

                self.change_pixmap.emit(frame)
                time.sleep(self.delay)
