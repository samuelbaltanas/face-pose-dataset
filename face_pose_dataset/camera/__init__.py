# flake8: noqa
from typing import Union

from .AstraCamera import *
from .WebCamera import *

CameraType = Union[VideoCamera, AstraCamera]

__all__ = ["AstraCamera", "WebCamera", "CameraType"]
