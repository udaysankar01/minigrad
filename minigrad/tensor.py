import numpy as np
from typing import Optional, Union, Callable, Set

class Tensor:
    """
    A class representing a multi-dimensional array (tensor) with support for
    automatic differentiation.

    The Tensor class holds data and gradients, and supports basic tensor
    operations like addition and multiplication. When `requires_grad` is set to
    True, the Tensor will accumulate gradients during backpropagation.

    Attributes:
        data (np.ndarray): The actual data stored in the tensor, converted to a NumPy array.
        requires_grad (bool): Indicates whether this tensor should calculate and accumulate
                              gradients during the backward pass.
        grad (np.ndarray or None): Stores the gradient of the tensor. Initially None.
        _backward (callable or None): Function to backpropagate gradients through the tensor.
        _prev (set): The set of parent tensors used to create this tensor.
        _op (str): Operation that created this tensor, used for debugging.
                    
    Example:
        >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    """
    def __init__(
            self,
            data: Union[float, list, np.ndarray],
            requires_grad: bool = False,
            _children: Set['Tensor'] = None,
            _op: str = '',
    ):
        """
        Initialize the Tensor object.

        Parameters:
            data (array-like): The initial values of the tensor. Can be a list, NumPy array, etc.
            requires_grad (bool, optional): If True, enables gradient tracking for this tensor.
                                  Defaults to False.
            _children (Set[Tensor], optional): Parent tensors that were used to compute this tensor.
                                               Defaults to empty set.
            _op (str, optional): Operation that created this tensor, used for debugging.
                                 Defaults to empty string.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None

        # Internal variables used for autograd graph construction
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = _children if _children is not None else set()
        self._op: str = _op

    def __repr__(self):
        return f"Tensor (Data={self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, _children={self,other}, _op='add')

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        return out

    def backward(self, grad: Optional['Tensor'] = None):
        """
        Computes the gradients by performing backprogation.
        Assumes the current tensor is a scalar (i.e., has a single value).

        Parameters:
            grad (Optional[Tensor]): The gradient of the current tensor with respect to some 
                                     scalar value. If `None`, and the tensor is a scalar, 
                                     a gradient of 1.0 is used. For non-scalar outputs, 
                                     `grad` must be provided.
        """
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs")
            grad = Tensor(1.0)
        
        self.grad = grad.data

        # Build topological order
        topo_order = []
        visited = set()

        def build_topo(tensor: Tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
            topo_order.append(tensor)
        
        build_topo(self)

        # backward pass
        for tensor in reversed(topo_order):
            tensor._backward()