from .tensor import Tensor

class MSELoss:
    """
    Mean Squared Error Loss. TODO: Add pow method
    """
    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        diff = predicted - target
        return (diff * diff).mean()