import numpy as np
import graphviz
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
        self._name: str = '' # for digraph

        # Internal variables used for autograd graph construction
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = _children if _children is not None else set()
        self._op: str = _op

    def __repr__(self):
        return f"Tensor(\n{self.data})\n"

    def __neg__(self):
        data = -self.data
        out = Tensor(data, self.requires_grad, _children={self}, _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad - out.grad if self.grad is not None else -out.grad
        
        out._backward = _backward
        return out

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

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, _children={self,other}, _op='mul')

        def _backward():
            if self.requires_grad:  
                grad = other.data * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad = other.grad + grad if other.grad is not None else grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, _children={self,other}, _op='div')

        def _backward():
            if self.requires_grad:
                grad = out.grad / other.data
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = (-self.data / (other.data ** 2)) * out.grad
                other.grad = other.grad + grad if other.grad is not None else grad
        out._backward = _backward
        return out
 
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, _children={self,other}, _op='matmul')

        def _backward():
            if self.requires_grad:
                grad = out.grad.dot(other.data.T)
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data.T.dot(out.grad)
                other.grad = other.grad + grad if other.grad is not None else grad
        out._backward = _backward
        return out

    def sum(self, axis: Optional[int] = None):
        """
        Returns a new tensor with the sum of the elements along the specified axis.
        If no axis is provided, sums over all elements.

        Parameters:
            axis (Optional[int]): The axis along which to sum. If None, sums over all elements.
        
        Returns:
            out (Tensor): A new tensor containing the summed values.
        
        Example:
            >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
            >>> b = a.sum(axis=0)
            >>> print(b.data) # Output: Tensor([4, 6])
        """
        data = self.data.sum(axis=axis)
        out = Tensor(data, self.requires_grad, _children={self}, _op='sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                print(grad)
                if axis is not None:
                    grad = np.expand_dims(grad, axis)
                    print(grad)
                grad = np.broadcast_to(grad, self.data.shape)
                print(grad)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out

    def relu(self):
        """
        Applies the ReLU (Rectified Linear Unit) function element-wise to the tensor.
        The ReLU function replaces all negative values in the tensor with zero.

        Returns:
            out (Tensor): A new tensor where all negative values are replaced by zero.

        Examples:
            >>> a = Tensor([-2.0, 3.0, -1.0, 4.0], requires_grad=True)
            >>> b = a.relu()
            >>> print(b.data)  # Output: [0. 3. 0. 4.]
        """
        data = np.maximum(0, self.data)
        out = Tensor(data, self.requires_grad, _children={self}, _op='relu')

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0).astype(self.data.dtype) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
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
            grad = Tensor(np.ones_like(self.data))

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

    # Visualization
    def graph(self, filename: Optional[str] =  None) -> graphviz.Digraph:
        """
        Generates a graphviz Digraph of the computational graph.
        """
        dot = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})
        visited = set()

        def add_nodes(tensor: 'Tensor'):
            if tensor not in visited:
                visited.add(tensor)

                tensor_id = str(id(tensor))
                tensor_label = self._tensor_label(tensor)
                dot.node(name=tensor_id, label=tensor_label, shape='record')

                if tensor._op:
                    op_id = tensor_id + tensor._op
                    op_label = self._op_label(tensor._op)
                    dot.node(name=op_id, label=op_label, shape='circle')
                    dot.edge(op_id, tensor_id)

                    for child in tensor._prev:
                        add_nodes(child)
                        child_id = str(id(child))
                        dot.edge(child_id, op_id)

        add_nodes(self)

        if filename:
            dot.render(filename=filename, directory="test", view=False)
        else:
            dot.render(filename="temp", directory="test", view=False)
        return dot
    
    def _tensor_label(self, tensor: 'Tensor') -> str:
        """
        Helper function to create a label for a tensor node.
        """
        label = ""
        if tensor._name:
            label += f"{tensor._name} | "
        label += f"shape: {tensor.data.shape}"
        return label

    def _op_label(self, op: str) -> str:
        """
        Helper function to create a label for a operation node.
        """
        label = ""
        op_dict = {
            "add" : "+",
            "mul" : "*",
            "neg" : "-",
            "sum" : "Σ",
            "div" : "/",
            "matmul" : "@",
            "relu" : "ReLU"
        }
        if op:
            if op in op_dict:
                label += f"{op_dict[op]}"
            else:
                label += op
        return label