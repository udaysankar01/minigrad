import cupy as cp
import numpy as np
import graphviz
from typing import Optional, Union, Callable, Set, Tuple

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
            requires_grad: bool = True,
            device: str = 'cpu',
            _children: Set['Tensor'] = None,
            _op: str = '',
            _name: str = ''
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
            _name (str, optional): An identifier for the tensor to be used in the graph visualization.
                                   Only for visualization purpose.
        """
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Optional[Union[np.ndarray, cp.ndarray]] = None
        self._name: str = _name # for digraph

        # convert array to appropriate array type (numpy or cupy)
        if not isinstance(data, (np.ndarray, cp.ndarray)):
            data = np.array(data, dtype=np.float32)
        
        if device == "cpu":
            data = self._to_cpu(data)
        elif device == "gpu":
            data = self._to_gpu(data)
        self.data = data

        # Internal variables used for autograd graph construction
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = _children if _children is not None else set()
        self._op: str = _op

    def _to_cpu(self, data)->np.ndarray:
        return cp.asnumpy(data) if isinstance(data, cp.ndarray) else data

    def _to_gpu(self, data)->cp.ndarray:
        return cp.array(data) if isinstance(data, np.ndarray) else data

    def to(self, device)->'Tensor':
        if self.device == device:
            return self
        
        # transfer data to specified device
        if device == "cpu":
            self.data = self._to_cpu(self.data)
            if self.grad is not None:
                self.grad = self._to_cpu(self.grad)
        elif device == "gpu":
            self.data = self._to_gpu(self.data)
            if self.grad is not None:
                self.grad = self._to_gpu(self.grad)
        else:
            raise ValueError("Device must be 'cpu' or 'gpu'")
        
        self.device = device
        return self

    def __repr__(self)->str:
        if self.device == 'cpu':
            return f"Tensor(\n{self.data})\n"
        else:
            return f"Tensor(\n{self.data}, device='{self.device}')"

    def __neg__(self)->'Tensor':
        data = -self.data
        out = Tensor(data, self.requires_grad, device=self.device, _children={self}, _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad = self.grad - out.grad if self.grad is not None else -out.grad
        
        out._backward = _backward
        return out

    def __add__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='add')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if self.data.shape != out.grad.shape:
                    grad = self._unbroadcast_grad(grad, self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = out.grad
                if other.data.shape != out.grad.shape:
                    grad = self._unbroadcast_grad(grad, other.data.shape)
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        return out

    def __sub__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        return self + (-other)

    def __mul__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='mul')

        def _backward():
            if self.requires_grad:  
                grad = other.data * out.grad
                if self.data.shape != out.grad.shape:
                    grad = self._unbroadcast_grad(grad, self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data * out.grad
                if other.data.shape != out.grad.shape:
                    grad = self._unbroadcast_grad(grad, other.data.shape)
                other.grad = other.grad + grad if other.grad is not None else grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='div')

        def _backward():
            if self.requires_grad:
                grad = out.grad / other.data
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = (-self.data / (other.data ** 2)) * out.grad
                other.grad = other.grad + grad if other.grad is not None else grad
        out._backward = _backward
        return out
 
    def __matmul__(self, other)->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='matmul')

        def _backward():
            if self.requires_grad:
                grad = out.grad.dot(other.data.T)
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data.T.dot(out.grad)
                other.grad = other.grad + grad if other.grad is not None else grad
        out._backward = _backward
        return out

    def sum(self, axis: Optional[int] = None)->'Tensor':
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
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='sum')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    grad = np.expand_dims(grad, axis)
                grad = np.broadcast_to(grad, self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out

    def mean(self, axis: Optional[int] = None)->'Tensor':
        """
        Compute the mean of the tensor along a specified axis.

        Parameters:
            axis (Optional[int]): The axis along which to compute the mean.
                        If None, computes the mean of all elements.
        
        Returns:
            out (Tensor): A new tensor representing the computed mean.
        
        Examples:
            >>> a = Tensor([[1, 2, 3], [3, 4, 5]], requires_grad=True)
            >>> b = a.mean(axis=0)
            >>> print(b.data)  # Output: [2. 3. 4.]
        """
        data = self.data.mean(axis=axis)
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='mean')

        def _backward():
            if self.requires_grad:
                grad = out.grad / np.prod(self.data.shape if axis is None else self.data.shape[axis])
                if axis is not None:
                    grad = np.expand_dims(grad, axis)
                grad = np.broadcast_to(grad, self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out

    def relu(self)->'Tensor':
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
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='relu')

        def _backward():
            if self.requires_grad:
                grad = (self.data > 0).astype(self.data.dtype) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out

    def T(self)->'Tensor':
        """
        Returns the transpose of the tensor.

        Returns:
            out (Tensor): A new tensor which is the transpose of the tensor.

        Examples:
            >>> A = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)
            >>> b = a.T()
            >>> print(b.data)  # Output: [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        """
        data = self.data.T
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='transpose')

        def _backward():
            if self.requires_grad:
                grad = out.grad.T
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out

    def reshape(self, *shape)->'Tensor':
        """
        Reshape the current tensor into a new shape withou changing its data.

        Parameters:
            *shape (int): Desired shape for the new tensor. Should be compatible
                        with the original tensor shape.
        
        Returns:
            out (Tensor): A new tensor with reshaped data. It shares same computational
                        graph for automatic differentiation.

        Examples:
            >>> x = Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
            >>> y = x.reshape(2, 2)
            >>> print(y) # [[1, 2], [3, 4]]
        """
        data = self.data.reshape(*shape)
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='reshape')

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out

    def backward(self, grad: Optional['Tensor'] = None):
        """
        Computes the gradients by performing backprogation.
        
        If the current tensor is scalar (i.e., a single value), the gradient defaults to 1.
        For non-scalar outputs, the gradient must be provided (since there's no implicit default).

        Parameters:
            grad (Optional[Tensor]): The gradient of the current tensor with respect to some 
                                     scalar value. If `None`, and the tensor is a scalar, 
                                     a gradient of 1.0 is used. For non-scalar outputs, 
                                     `grad` must be provided.
        """
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.data.size == 1:
                grad = Tensor(1.0, device=self.device)
            else:
                raise ValueError("grad must be provided for non-scalar outputs.")

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
    
    def zero_grad(self):
        self.grad = None
        for child in self._prev:
            child.zero_grad()

    # Slicing
    def __getitem__(self, idx)->'Tensor':
        data = self.data[idx]
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='slice')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    # Helper methods for convinience
    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        data = np.zeros(shape, dtype=np.float32)
        return Tensor(data, requires_grad, device)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        data = np.ones(shape, dtype=np.float32)
        return Tensor(data, requires_grad, device)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        data = np.random.randn(*shape).astype(np.float32)
        return Tensor(data, requires_grad, device)
    
    @staticmethod
    def arange(start: int, end: int, step: int = 1, requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        data = np.arange(start, end, step, dtype=np.float32)
        return Tensor(data, requires_grad, device)

    def _unbroadcast_grad(self, grad, shape):
        """
        Adjusts the gradient to account for broadcasting during forward pass.

        Parameters:
            grad (np.ndarray): The gradient from the next layer (out.grad).
            shape (Tuple[int, ...]): The shape of the original tensor before broadcasting.
        
        Returns:
            grad (np.ndarray): The adjusted gradient that matches the original tensor's shape. 
        """
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for axis, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
            if shape_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad
    
    ### Implementing __r*__ method for operations with scalars
    def __radd__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        return self + other

    def __rsub__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        return (-self) + other
    
    def __rmul__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        return self * other
    
    def __rtruediv__(self, other: Union[float, int, list, np.ndarray])->'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other / self
    
    # TODO: Overload comparison operators for tensors (element-wise)

    ### Visualization
    def graph(self, filename: Optional[str] =  None, show_data: bool = False, show_grad: bool = False) -> graphviz.Digraph:
        """
        Generates a graphviz Digraph of the computational graph.

        Parameters:
            filename (optional, str): filename to store the generated graph
            show_data (bool): Boolean to determine whether to show the data in the graph
            show_grad (bool): Boolean to detemine whether to show the gradient in the graph
        
        Returns:
            dot (graphviz.Digraph): Digraph object from graphviz
        """
        dot = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})
        visited = set()

        def add_nodes(tensor: 'Tensor'):
            if tensor not in visited:
                visited.add(tensor)

                tensor_id = str(id(tensor))
                tensor_label = self._tensor_label(tensor, show_data, show_grad)
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
            dot.render(filename="example", directory="test", view=False)
        return dot
    
    def _tensor_label(self, tensor: 'Tensor', show_data=False, show_grad=False) -> str:
        """
        Helper function to create a label for a tensor node.
        """
        label = ""
        if tensor._name:
            label += f"{tensor._name} | "
        label += f"shape: {tensor.data.shape}"
        if show_data:
            label += f"| data: {tensor.data}"
        if show_grad:
            label += f"| grad: {tensor.grad}"
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