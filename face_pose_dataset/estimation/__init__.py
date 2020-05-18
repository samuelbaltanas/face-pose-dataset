# flake8: noqa

from .mtcnn import *
from .base import *
from .ensemble import *
from .interface import *

ESTIMATORS = {
    "ddfa": DdfaEstimator,
    "fsa": FSAEstimator,
    "hope": HopenetEstimator,
    "multi": AverageEstimator,
}
