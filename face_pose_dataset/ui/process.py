import time

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from face_pose_dataset import camera
from face_pose_dataset.third_party import FSA_net

class EstimationThread(QtCore.QThread):
    change_pixmap = QtCore.Signal(np.ndarray)
    change_pose = QtCore.Signal(np.ndarray)

    def __init__(self, width=640, height=480, delay=1 / 20):
        super().__init__()
        self.width = width
        self.height = height
        self.delay = delay

        # Face pose estimation
        self.detector = FSA_net.SSDDetector()
        self.estimator = FSA_net.FSAEstimator()
        pass

    def run(self):
        with camera.AstraCamera() as cam:
            while True:
                frame = cam.read_rgb()

                # TODO. Compute frames.

                res = self.detector.run(frame)

                det = FSA_net.extract_faces(
                    frame, res, self.estimator.img_size, threshold=0.8
                )

                if det is not None:
                    ang = self.estimator.run(det)

                    self.change_pose.emit(ang[0])

                self.change_pixmap.emit(frame)
                # TODO. Change sleep to use elapsed time.
                # time.sleep(self.delay)
