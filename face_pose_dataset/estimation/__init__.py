# flake8: noqa

from .base import *
from .mtcnn import *
from .ensemble import *
from .interface import *


ESTIMATORS = {
    "ddfa": DdfaEstimator,
    "fsa": FSAEstimator,
    "hope": HopenetEstimator,
    "multi": AverageEstimator,
}
