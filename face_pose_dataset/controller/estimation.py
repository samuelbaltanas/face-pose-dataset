import logging
import time

import numpy as np
from PySide2 import QtCore

from face_pose_dataset import camera, estimation
from face_pose_dataset.core import EstimationData, Image

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
        self.runs = True

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
        self.estimator = estimation.AverageEstimator()
        # self.estimator = estimation.SklearnEstimator()
        # self.estimator = estimation.NnetWrapper(checkpoint=99, out_loss=1)

    def run(self):
        if self.camera is None:
            self.cond.wait(self.mutex)

        # DONE Fix error with opencv camera
        # SOL. Error in waiting condition for thread (wake before camera is set).

        # TODO: Brainstorm for solutions (slow fps on cpu)
        with self.camera() as cam:
            while self.runs:
                bbox = None
                start_time = time.time()

                frame, depth = cam.read_both()
                self.video_feed.emit(frame)

                if self.is_paused:
                    self.cond.wait(self.mutex)
                    continue

                # DONE. Compute frames.
                res = self.detector.run(frame, threshold=0.8)

                if res is not None:
                    # DONE. Improve selection process on multiple faces. Use centermost bbox.
                    if len(res[0]) > 1:
                        bboxes = np.array(res[0])[:, :4]
                        frame_center = np.array(frame.shape)[:2] // 2
                        boxes_center = np.abs(bboxes[:, ::2] - bboxes[:, 1::2]) // 2
                        idx = np.argmin(
                            np.sum(np.abs(boxes_center - frame_center), axis=1)
                        )

                        bbox = bboxes[idx]
                    else:
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

    def toggle_pause(self,):
        self._paused = not self._paused
        self.try_wake()

    @QtCore.Slot(bool)
    def set_pause(self, b: bool):
        self._paused = b
        self.try_wake()

    @QtCore.Slot(bool)
    def set_stop(self, b: bool):
        self.runs = not b

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
