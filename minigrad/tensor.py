import numpy as np
from typing import Optional, Union, Callable, Set

class Tensor:
    def __init__(
            self,
            data: Union[float, list, np.ndarray],
            requires_grad: bool = False
    ):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None

        # Internal variables used for autograd graph construction
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set['Tensor'] = {}
        self._op: str = ""

    def __repr__(self):
        return f"Tensor (Data={self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: Optional['Tensor'] = None):
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