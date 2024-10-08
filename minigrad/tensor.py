import cupy as cp
import numpy as np
import graphviz
from typing import Optional, Union, Callable, Set, Tuple

TensorLike = Union[float, int, list, np.ndarray, cp.ndarray, 'Tensor']

class Tensor:
    """
    A class representing a multi-dimensional array (tensor) with support for
    automatic differentiation.

    The Tensor class holds data and gradients, and supports basic tensor
    operations like addition and multiplication. When `requires_grad` is set to
    True, the Tensor will accumulate gradients during backpropagation.

    Attributes:
        data (np.ndarray or cp.ndarray): The actual data stored in the tensor, converted to a NumPy array.
        requires_grad (bool): Indicates whether this tensor should calculate and accumulate
                gradients during the backward pass.
        device (str): The device where the tensor data is stored ('cpu' or 'gpu'). Defaults to 'cpu'.
        grad (np.ndarray or None): Stores the gradient of the tensor. Initially None.
        _backward (callable or None): Function to backpropagate gradients through the tensor.
        _prev (set): The set of parent tensors used to create this tensor.
        _op (str): Operation that created this tensor, used for debugging.
                    
    Example:
        >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    """
    def __init__(
            self,
            data: TensorLike,
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
            device (str): The device where the tensor data is stored  ('cpu' or 'gpu'). 
                    Defaults to 'cpu'.
            _children (Set[Tensor], optional): Parent tensors that were used to compute this tensor.
                    Defaults to empty set.
            _op (str, optional): Operation that created this tensor, used for debugging.
                    Defaults to empty string.
            _name (str, optional): An identifier for the tensor to be used in the graph visualization.
                    Only for visualization purpose.
        """
        self.device: str = device
        self.requires_grad: bool = requires_grad
        self.grad: Optional[Union[np.ndarray, cp.ndarray]] = None
        self._name: str = _name # for digraph

        # convert array to appropriate array type (numpy or cupy) -- redundant for gpu (fix later)
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

    def _to_cpu(self, data: Union[cp.ndarray, np.ndarray])->np.ndarray:
        return cp.asnumpy(data) if isinstance(data, cp.ndarray) else data

    def _to_gpu(self, data: Union[np.ndarray, cp.ndarray])->cp.ndarray:
        return cp.array(data) if isinstance(data, np.ndarray) else data

    def to(self, device: str)->'Tensor':
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
        
    def _apply_grad(self, tensor: 'Tensor', grad: Union[np.ndarray, cp.ndarray]):
        if tensor.requires_grad:
            if tensor.data.shape != grad.shape:
                grad = self._unbroadcast_grad(grad, tensor.data.shape)
            tensor.grad = tensor.grad + grad if tensor.grad is not None else grad

    def _backward_fn(self, out: 'Tensor', self_grad_fn: Callable, other_grad_fn: Optional[Callable]=None):
        """
        Unary and Binary operation backward pass.
        """
        def _backward():
            if self_grad_fn and self.requires_grad:
                self_grad_fn(out.grad)

            if other_grad_fn:
                other_grad_fn(out.grad)
        return _backward

    def __neg__(self)->'Tensor':
        data = -self.data
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op="neg")

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, -grad)
        )
        return out
    
    def __pow__(self, power: int)-> 'Tensor':
        data = self.data ** power
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op="pow")

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad * (power * (self.data ** (power - 1))))
        )
        return out

    def __add__(self, other: TensorLike)->'Tensor':
        other: 'Tensor' = _ensure_tensor(other, self.device)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self, other}, _op='add')

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad),
            other_grad_fn=lambda grad: self._apply_grad(other, grad)
        )
        return out

    def __sub__(self, other: Union[float, int, list, np.ndarray, cp.ndarray, 'Tensor'])->'Tensor':
        return self + (-other)
    
    def __mul__(self, other: Union[float, int, list, np.ndarray, cp.ndarray, 'Tensor'])->'Tensor':
        other: 'Tensor' = _ensure_tensor(other, self.device)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='mul')

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad * other.data),
            other_grad_fn=lambda grad: self._apply_grad(other, grad * self.data)
        )
        return out

    def __truediv__(self, other: Union[float, int, list, np.ndarray, cp.ndarray, 'Tensor'])->'Tensor':
        other: 'Tensor' = _ensure_tensor(other, self.device)
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='div')

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad / other.data),
            other_grad_fn=lambda grad: self._apply_grad(other, (-self.data / (other.data ** 2)) * grad)
        )
        return out
 
    def __matmul__(self, other: Union[np.ndarray, cp.ndarray, 'Tensor'])->'Tensor':
        other: 'Tensor' = _ensure_tensor(other, self.device)
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad, self.device, _children={self,other}, _op='matmul')

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad.dot(other.data.T)),
            other_grad_fn=lambda grad: self._apply_grad(other, self.data.T.dot(grad))
        )
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
        xp = cp if self.device == "gpu" else np
        data = xp.sum(self.data, axis=axis)
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='sum')

        def grad_fn(out_grad, axis, original_shape):
            grad = out_grad
            if axis is not None:
                grad = xp.expand_dims(grad, axis)
            grad = xp.broadcast_to(grad, original_shape)
            return grad
        
        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad_fn(grad, axis, self.data.shape))        
        )
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
        xp = cp if self.device == "gpu" else np
        data = xp.mean(self.data, axis=axis)
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='mean')

        def grad_fn(out_grad, axis, original_shape):
            grad = out_grad / xp.prod(xp.array(original_shape if axis is None else original_shape[axis]))
            if axis is not None:
                grad = xp.expand_dims(grad, axis)
            grad = xp.broadcast_to(grad, original_shape)
            return grad

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad_fn(grad, axis, self.data.shape))
        )
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
        xp = cp if self.device == "gpu" else np
        data = xp.maximum(0, self.data)
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='relu')
        
        def grad_fn(out_grad, data, dtype):
            grad = (data > 0).astype(dtype) * out_grad
            return grad
        
        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad_fn(grad, self.data, self.data.dtype))
        )
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
        
        out._backward = self._backward_fn(
            out, 
            self_grad_fn=lambda grad: self._apply_grad(self, grad.T)
        )
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
        
        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad.reshape(self.data.shape))
        )
        return out

    # Slicing
    def __getitem__(self, idx)->'Tensor':
        xp = cp if self.device == "gpu" else np
        data = self.data[idx]
        out = Tensor(data, self.requires_grad, self.device, _children={self}, _op='slice')
        
        def grad_fn(out_grad, data):
            grad = xp.zeros_like(data)
            grad[idx] = out_grad
            return grad

        out._backward = self._backward_fn(
            out,
            self_grad_fn=lambda grad: self._apply_grad(self, grad_fn(grad, self.data))
        )
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
        active_visits = set()

        def build_topo(tensor: Tensor):
            if tensor in active_visits:
                raise RuntimeError(f"Cycle detected in the computation graph at tensor: {tensor}")
            if tensor not in visited:
                active_visits.add(tensor)
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
            active_visits.remove(tensor)
            topo_order.append(tensor)
        
        build_topo(self)

        # backward pass
        for tensor in reversed(topo_order):
            tensor._backward()
    
    def zero_grad(self):
        self.grad = None
        for child in self._prev:
            child.zero_grad()
    
    # Helper methods for convinience
    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        xp = cp if device == "gpu" else np
        data = xp.zeros(shape, dtype=xp.float32)
        return Tensor(data, requires_grad, device)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        xp = cp if device == "gpu" else np
        data = xp.ones(shape, dtype=xp.float32)
        return Tensor(data, requires_grad, device)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        xp = cp if device == "gpu" else np
        data = xp.random.randn(*shape).astype(xp.float32)
        return Tensor(data, requires_grad, device)
    
    @staticmethod
    def arange(start: int, end: int, step: int = 1, requires_grad: bool = True, device: str = 'cpu')->'Tensor':
        xp = cp if device == "gpu" else np
        data = xp.arange(start, end, step, dtype=xp.float32)
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
    def __radd__(self, other: TensorLike)->'Tensor':
        return self + other

    def __rsub__(self, other: TensorLike)->'Tensor':
        return (-self) + other
    
    def __rmul__(self, other: TensorLike)->'Tensor':
        return self * other
    
    def __rtruediv__(self, other: TensorLike)->'Tensor':
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
    
def _ensure_tensor(input: TensorLike, device: str) -> 'Tensor':
    input: 'Tensor' = input if isinstance(input, Tensor) else Tensor(input, device=device)
    return input