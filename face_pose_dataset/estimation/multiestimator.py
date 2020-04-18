from typing import Tuple

import numpy as np

from face_pose_dataset import core
from face_pose_dataset.estimation import ddfa, fsanet, hopenet, interface

__all__ = ["MultiEstimator"]


class MultiEstimator(interface.Estimator):
    def __init__(self):
        self.hope = hopenet.HopenetEstimator()
        self.ddfa = ddfa.DdfaEstimator()
        self.fsa = fsanet.FSAEstimator()

    def preprocess_image(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        im1 = self.hope.preprocess_image(frame, bbox)
        im2 = self.ddfa.preprocess_image(frame, bbox)
        im3 = self.fsa.preprocess_image(frame, bbox)

        return im1, im2, im3

    def run(
        self, input_images: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> core.Angle:
        res1 = self.hope(input_images[0])
        res2 = self.ddfa(input_images[1])
        res3 = self.fsa(input_images[2])

        r = np.array([res1, res2, res3]).mean(axis=0)
        res = core.Angle(*r)
        return res
