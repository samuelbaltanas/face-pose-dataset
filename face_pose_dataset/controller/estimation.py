import logging
import time

import cv2
import tensorflow as tf
from PySide2 import QtCore

from face_pose_dataset import camera, estimation
from face_pose_dataset.core import EstimationData, Image

# TODO Move this piece of code to a startup function.
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


COMPATIBLE_CAMERAS = {
    "astra": camera.AstraCamera,
    "webcam": camera.VideoCamera,
}


class EstimationThread(QtCore.QThread):
    video_feed = QtCore.Signal(Image)
    result_signal = QtCore.Signal(EstimationData)

    def __init__(self, width=640, height=480, delay=1 / 20):
        super().__init__()
        self.width = width
        self.height = height
        self.delay = delay

        self.mutex = QtCore.QMutex()
        self.cond = QtCore.QWaitCondition()
        self._paused = True
        self.camera = None

        # Face pose estimation
        # self.detector = estimation.SSDDetector()
        self.detector = estimation.MTCNN()
        # self.estimator = estimation.HopenetEstimator()
        # self.estimator = estimation.DdfaEstimator()
        # self.estimator = estimation.FSAEstimator()
        self.estimator = estimation.MultiEstimator()

    def run(self):
        if self.camera is None:
            self.cond.wait(self.mutex)

        # DONE Fix error with opencv camera
        # SOL. Error in waiting condition for thread (wake before camera is set).

        # TODO: Brainstorm for solutions (slow fps on cpu)
        with self.camera() as cam:
            while True:
                bbox = None
                start_time = time.time()

                frame, depth = cam.read_both()
                self.video_feed.emit(frame)

                # TODO: RGB/BGR discrepancy between Astra and opencv cameras
                frame = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_BGR2RGB)

                if self.is_paused:
                    self.cond.wait(self.mutex)
                    continue

                # DONE. Compute frames.

                res = self.detector.run(frame, threshold=0.8)

                if res is not None:
                    # TODO. Improve selection process on multiple faces.
                    bbox = res[0][0][:4]

                    det = self.estimator.preprocess_image(frame, bbox)
                    ang = self.estimator.run(det)

                    logging.debug("Sending angle %s .", ang)

                    est = EstimationData(box=bbox, angle=ang, rgb=frame, depth=depth)
                    self.result_signal.emit(est)

                # DONE. Change sleep to use elapsed time.
                elapsed_time = time.time() - start_time

                sleep_time = self.delay - elapsed_time
                if sleep_time > 0.0:
                    logging.debug("Estimation thread sleep for: %s", sleep_time)
                    time.sleep(sleep_time)

    def __del__(self):
        self.terminate()

    def toggle_pause(self,):
        self._paused = not self._paused
        self.try_wake()

    @QtCore.Slot(bool)
    def set_pause(self, b: bool):
        self._paused = b
        self.try_wake()

    @QtCore.Slot(str)
    def init_camera(self, camera_string):
        """ Start camera according to form information. """
        self._paused = False
        self.camera = COMPATIBLE_CAMERAS[camera_string]
        self.try_wake()

    @property
    def is_paused(self):
        """ Check if thread should be paused """
        return self._paused or self.camera is None

    def try_wake(self):
        """ Performs sanity check and wakes up thread if needed """
        if not self.is_paused:
            self.cond.wakeAll()
