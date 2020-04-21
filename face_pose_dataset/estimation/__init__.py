# flake8: noqa

from .ddfa import *
from .fsanet import *
from .hopenet import *
from .interface import *
from .mtcnn import *
from .multiestimator import *

ESTIMATORS = {
    "ddfa": DdfaEstimator,
    "fsanet": FSAEstimator,
    "hopenet": HopenetEstimator,
    "multi": MultiEstimator,
}
