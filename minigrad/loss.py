from .tensor import Tensor

class MSELoss:
    """
    Mean Squared Error Loss. TODO: Add pow method
    """
    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        squared_diff = (predicted - target) ** 2
        return squared_diff.mean()