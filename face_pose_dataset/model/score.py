import logging
import random
from typing import Tuple

import numpy as np
from PySide2 import QtCore

from face_pose_dataset import core
from face_pose_dataset.core import Angle

DEFAULT_YAW_RANGE = -90.0, 90.0  # horizontal
DEFAULT_PITCH_RANGE = -40.0, 40.0  # vertical

YAW_DIR = -1
PITCH_DIR = -1


class ScoreModel(QtCore.QObject):
    change_score = QtCore.Signal(np.ndarray)

    def __init__(
        self,
        dimensions: tuple,  # [vdim, hdim]
        yaw_range=DEFAULT_YAW_RANGE,  # horizontal
        pitch_range=DEFAULT_PITCH_RANGE,  # vertical
    ):
        super().__init__()
        self._min_yaw = yaw_range[0]
        self._yaw_inc = (yaw_range[1] - yaw_range[0]) / dimensions[1]

        self._min_pitch = pitch_range[1]
        self._pitch_inc = -(pitch_range[1] - pitch_range[0]) / dimensions[0]

        self._score_max = np.sqrt(self._yaw_inc ** 2 + self._pitch_inc ** 2) / 2

        self.scores: np.ndarray[float] = np.full(dimensions, self._score_max)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.scores.shape

    def add_angle(self, angle: core.Angle) -> bool:
        pose = self.locate_angle(angle)

        if 0 <= pose[0] < self.shape[0] and 0 <= pose[1] < self.shape[1]:
            target = self.convert_position(pose)
            score = np.sqrt(
                (angle.yaw - target.yaw) ** 2 + (angle.pitch - target.pitch) ** 2
            )

            if score < self[pose[1], pose[0]]:
                self[pose[1], pose[0]] = score
                logging.debug("Image saved at: %s", angle)
                self.change_score.emit(self.scores.data)
                return True

        return False

    def locate_angle(self, angle: core.Angle) -> Tuple[int, int]:
        """ Find which cell of score_matrix is the nearest.

        REQ: Find where a new angle should be inserted.

        Args:
            angle: Angle to query.
        """

        y = int((angle.pitch - self._min_pitch) // self._pitch_inc)
        x = int((angle.yaw - self._min_yaw) // self._yaw_inc)

        return x, y

    def iloc(self, angle: Angle) -> Tuple[float, float]:
        """
        Args:
            angle: Angle to query.
        """

        y = (angle.pitch - self._min_pitch) / self._pitch_inc - 0.5
        x = (angle.yaw - self._min_yaw) / self._yaw_inc - 0.5

        return x, y

    def convert_position(self, position: Tuple[int, int]) -> core.Angle:
        """ Find the angle corresponding to the center of a cell.

        REQ: Compute the distance score.
        """

        pitch = (position[1] + 0.5) * self._pitch_inc + self._min_pitch
        yaw = (position[0] + 0.5) * self._yaw_inc + self._min_yaw

        return core.Angle(0.0, pitch, yaw)

    @property
    def y_range(self) -> np.ndarray:
        return np.arange(
            self._min_pitch + 0.5 * self._pitch_inc,
            self._min_pitch + 0.5 * self._pitch_inc + self._pitch_inc * self.shape[0],
            self._pitch_inc,
        )

    @property
    def x_range(self) -> np.ndarray:
        return np.arange(
            self._min_yaw + 0.5 * self._yaw_inc,
            self._min_yaw + 0.5 * self._yaw_inc + self._yaw_inc * self.shape[1],
            self._yaw_inc,
        )

    @property
    def z_range(self) -> Tuple[float, float]:
        return 0.0, self._score_max

    def random_idx(self):
        return (
            random.randint(0, self.shape[0] - 1),
            random.randint(0, self.shape[1] - 1),
        )

    def __getitem__(self, item: Tuple[int, int]) -> float:
        return self.scores[item]

    def __setitem__(self, key: Tuple[int, int], value: float):
        self.scores[key] = value
