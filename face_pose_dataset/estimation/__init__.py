# flake8: noqa

from .base import *
from .ensemble import *
from .interface import *
from .mtcnn import *

ESTIMATORS = {
    "ddfa": DdfaEstimator,
    "fsa": FSAEstimator,
    "hope": HopenetEstimator,
    "multi": AverageEstimator,
}
