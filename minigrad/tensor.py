"""
A Tensor is a multi-dimensional array, similar to NumPy arrays,
that stores data and computes gradients for automatic differentiation.

It supports basic operations like addition, multiplication,
and matrix multplication, while tracking dependencies for backpropagation.

The Tensor class is the core building block for constructing neural networks
and implementing automatic differentiation (autograd).
"""

import numpy as np

class Tensor:

    def __init__(self, data, requires_grad=False):
        """
        Initialize a tensor with data and optional gradient tracking.

        Parameters
        ----------
        data (array-like): The data to store in the tensor.
        requires_grad (bool): Whether to track gradients for this tensor.
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None                    # Gradient for backpropagation
        self._backward = None               # Backward function for autograd
        self._prev = set()                  # Tensors this tensor depends on
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

a = np.array([1, 2, 3, 3, 4])
tensor = Tensor(a, requires_grad=True)
print(tensor)