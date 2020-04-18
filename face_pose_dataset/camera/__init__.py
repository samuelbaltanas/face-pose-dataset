# flake8: noqa
from typing import Union

from .astra import *
from .usb import *

CameraType = Union[VideoCamera, AstraCamera]

__all__ = ["AstraCamera", "usb.py", "CameraType"]
