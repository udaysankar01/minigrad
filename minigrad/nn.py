import math
import numpy as np
from typing import List
from .tensor import Tensor

class Layer:
    """
    Base class for all neural network layers.
    """
    def __init__(self):
        self.params = []
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def paramters(self):
        return self.params

class Linear(Layer):
    """
    A fully connected layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor(np.random.uniform(-bound, bound, (out_features, in_features)), requires_grad=True)
        self.bias = None
        self.params = [self.weight]
        if bias:
            self.bias = Tensor(np.random.uniform(-bound, bound, out_features), requires_grad=True)
            self.params.append(self.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.T()
        if self.bias is not None:
            output = output + self.bias
        return output
