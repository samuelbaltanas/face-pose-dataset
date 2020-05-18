import logging
import sys
import time
import traceback

import numpy as np
from PySide2 import QtCore

from face_pose_dataset import camera, estimation
from face_pose_dataset.core import EstimationData, Image

COMPATIBLE_CAMERAS = {
    "astra": camera.AstraCamera,
    "webcam": camera.VideoCamera,
}

class EstimationWorker(QtCore.QObject):
    result_signal = QtCore.Signal(EstimationData)
    signal_done = QtCore.Signal()
    signal_prepared = QtCore.Signal()

    def __init__(self, width=640, height=480, gpu=0):
        super().__init__()
        self.width = width
        self.height = height

        self.detector = None
        self.estimator = None

        self.gpu = gpu

    @QtCore.Slot()
    def start_est(self):
        try:
            logging.debug("[EST] Loading estimators")
            # Face pose estimation
            # self.detector = estimation.SSDDetector()
            self.detector = estimation.MTCNN()
            # self.estimator = estimation.HopenetEstimator()
            # self.estimator = estimation.DdfaEstimator()
            # self.estimator = estimation.FSAEstimator()
            self.estimator = estimation.AverageEstimator(gpu=self.gpu)
            # self.estimator = estimation.SklearnEstimator()
            # self.estimator = estimation.NnetWrapper(checkpoint=99, out_loss=1)
            self.signal_prepared.emit()
            logging.debug("[EST] Estimators loaded")
        except Exception as e:
            track = traceback.format_exc()
            logging.error(track)
            sys.exit(-1)

    @QtCore.Slot(tuple)
    def run(self, frames: tuple):
        try:
            frame, depth = frames
            logging.debug("[EST] Receiving frames")
            # DONE. Compute frames.
            res = self.detector.run(frame, threshold=0.8)

            if res is not None:
                # DONE. Improve selection process on multiple faces. Use centermost bbox.
                if len(res[0]) > 1:
                    bboxes = np.array(res[0])[:, :4]
                    frame_center = np.array(frame.shape)[:2][::-1] // 2
                    boxes_center = bboxes[:, :2] + bboxes[:, 2:]// 2
                    idx = np.argmin(
                        np.sum(np.abs(boxes_center - frame_center), axis=1)
                    )

                    bbox = bboxes[idx]
                else:
                    bbox = res[0][0][:4]

                det = self.estimator.preprocess_image(frame, bbox)
                ang = self.estimator.run(det)

                logging.debug("[EST] Sending angle %s .", ang)

                est = EstimationData(box=bbox, angle=ang, rgb=frame, depth=depth)
                self.result_signal.emit(est)
            self.signal_done.emit()
        except Exception as e:
            track = traceback.format_exc()
            logging.error(track)
            sys.exit(-1)



class EstimationThread(QtCore.QObject):
    video_feed = QtCore.Signal(np.ndarray)
    est_feed = QtCore.Signal(tuple)
    signal_start = QtCore.Signal()

    def __init__(self, delay=1 / 20, gpu=0):
        super().__init__()
        self.delay = delay
        self.gpu = gpu
        self.is_prepared = False
        self._paused = False
        self._is_terminated = False

        self.camera = None

        self.worker = EstimationWorker(gpu=gpu)

        self.estimation_thread = QtCore.QThread()
        self.worker.moveToThread(self.estimation_thread)

        self.worker.signal_done.connect(self.send_frame)
        self.signal_start.connect(self.worker.start_est)
        self.est_feed.connect(self.worker.run)
        self.worker.signal_prepared.connect(self.set_prepared)

        self.estimation_thread.start()

        self.signal_start.emit()

    @QtCore.Slot()
    def set_prepared(self):
        self.is_prepared = True
        if self.camera is not None:
            self.send_frame()

    @QtCore.Slot()
    def send_frame(self):
        if self._is_terminated:
            logging.info("[CAMERA] Camera terminated.")
            if self.camera is not None:
                self.camera.__exit__()
            self.estimation_thread.quit()
        elif not self._paused:
            logging.debug("[CAMERA] Frame sent.")

            frame, depth = self.camera.read_both()
            self.est_feed.emit((frame, depth))
            self.video_feed.emit(frame)

    def toggle_pause(self,):
        self._paused = not self._paused
        logging.info("[CAMERA] Pause toggled. Is paused: %s", self._paused)
        self.send_frame()

    @QtCore.Slot(bool)
    def set_pause(self, b: bool = True):
        if self._paused and b:
            self._paused = b
            self.send_frame()
        else:
            self._paused = b

        logging.info("[CAMERA] Pause set to: %s", self._paused)


    @QtCore.Slot(bool)
    def set_stop(self, b: bool = True):
        logging.info("[CAMERA] Camera receives termination signal.")
        self._is_terminated = True

    @QtCore.Slot(str)
    def init_camera(self, camera_string):
        """ Start camera according to form information. """
        self._paused = False
        self.camera = COMPATIBLE_CAMERAS[camera_string]().__enter__()
        if self.is_prepared:
            self.send_frame()
