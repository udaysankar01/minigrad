"""
minigrad
========
"""
__version__ = "0.0.1"

# expose core components at the package level
from .tensor import Tensor
from .optim import SGD
from .loss import MSELoss