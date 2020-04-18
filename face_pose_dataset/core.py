from typing import NamedTuple, Optional, Tuple

import numpy as np

Image = np.ndarray
Position = Tuple[int, int]


class Angle(NamedTuple):
    """ Wrapper over tuple for use in angles. Format: roll, pitch, yaw. """

    roll: float
    pitch: float
    yaw: float


class EstimationData(NamedTuple):
    box: np.ndarray
    angle: Angle
    rgb: np.ndarray
    depth: Optional[np.ndarray]
