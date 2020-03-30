import random
from typing import Iterable, Tuple

import numpy as np

from face_pose_dataset import core

DEFAULT_YAW_RANGE = -90.0, 90.0
DEFAULT_PITCH_RANGE = -180.0, 180.0


class ScoreMatrix:
    def __init__(
        self,
        dimensions: tuple,
        yaw_range=DEFAULT_YAW_RANGE,
        pitch_range=DEFAULT_PITCH_RANGE,
    ):
        self._min_yaw = yaw_range[0]
        self._yaw_inc = (yaw_range[1] - yaw_range[0]) / dimensions[1]
        self._min_pitch = pitch_range[0]
        self._pitch_inc = (pitch_range[1] - pitch_range[0]) / dimensions[0]

        self._score_max = np.sqrt(self._yaw_inc ** 2 + self._pitch_inc ** 2) / 2

        self.scores: np.ndarray[float] = np.full(dimensions, self._score_max)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.scores.shape

    def locate_angle(self, angle: core.Angle) -> Tuple[int, int]:
        """ Find which cell of score_matrix is the nearest.

        REQ: Find where a new angle should be inserted.

        Args:
            angle: Angle to query.
        """

        _, pitch, yaw = angle

        x = int((pitch - self._min_pitch) // self._pitch_inc)
        y = int((yaw - self._min_yaw) // self._yaw_inc)

        return x, y

    def convert_position(self, position: Tuple[int, int]) -> core.Angle:
        """ Find the angle corresponding to the center of a cell.

        REQ: Compute the distance score.
        """

        pitch = (position[0] + 0.5) * self._pitch_inc + self._min_pitch
        yaw = (position[1] + 0.5) * self._yaw_inc + self._min_yaw

        return core.Angle(0.0, pitch, yaw)

    @property
    def y_range(self) -> Iterable[float]:
        return np.arange(
            self._min_yaw + 0.5 * self._yaw_inc,
            self._min_yaw + 0.5 * self._yaw_inc + self._yaw_inc * self.shape[0],
            self._yaw_inc,
        )

    @property
    def x_range(self) -> Iterable[float]:
        return np.arange(
            self._min_pitch + 0.5 * self._pitch_inc,
            self._min_pitch + 0.5 * self._pitch_inc + self._pitch_inc * self.shape[1],
            self._pitch_inc,
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
