from typing import NamedTuple, Tuple

Position = Tuple[int, int]


class Angle(NamedTuple):
    """ Wrapper over tuple for use in angles. Format: roll, pitch, yaw. """

    roll: float
    pitch: float
    yaw: float
