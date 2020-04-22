from typing import Tuple

import numpy as np

from face_pose_dataset import core
from face_pose_dataset.estimation import ddfa, fsanet, hopenet, interface

__all__ = ["MultiEstimator"]


class MultiEstimator(interface.Estimator):
    def __init__(self):
        self.estimators = (
            hopenet.HopenetEstimator(),
            ddfa.DdfaEstimator(),
            fsanet.FSAEstimator(),
        )

    def preprocess_image(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        res = [est.preprocess_image(frame, bbox) for est in self.estimators]

        return tuple(res)

    def run(self, input_images: Tuple[np.ndarray, ...]) -> core.Angle:
        r = [est(frame) for est, frame in zip(self.estimators, input_images)]
        r = np.array(r).mean(axis=0)

        # TODO. Use error on validation datasets as weights for mean
        res = core.Angle(*r)
        return res
