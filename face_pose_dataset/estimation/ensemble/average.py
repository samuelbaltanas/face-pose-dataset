from typing import Tuple, Union

import numpy as np

from face_pose_dataset import core
from face_pose_dataset.estimation import interface
from face_pose_dataset.estimation.base import ddfa, fsanet, hopenet

__all__ = ["AverageEstimator"]


class AverageEstimator(interface.Estimator):
    def __init__(self, activation="mean"):
        self.estimators = (
            hopenet.HopenetEstimator(),
            ddfa.DdfaEstimator(),
            fsanet.FSAEstimator(),
        )

        self.activation = activation

    def preprocess_image(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        res = [est.preprocess_image(frame.copy(), bbox) for est in self.estimators]

        return tuple(res)

    def run(
        self, input_images: Tuple[np.ndarray, ...]
    ) -> Union[core.Angle, np.ndarray]:
        r = [est(frame) for est, frame in zip(self.estimators, input_images)]

        if self.activation == "mean":
            r = np.array(r).mean(axis=0)

            res = core.Angle(*r)
            return res
        elif self.activation == "concat":
            r = np.concatenate(r).reshape((1, -1))
            assert r.shape[0] == 1 and r.shape[1] == 9
            return r
