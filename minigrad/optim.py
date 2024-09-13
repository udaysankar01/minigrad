class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, params, lr: float = 0.01):
        self.params = params
        self.lr = lr
    
    def step(self):
        """
        Performs a gradient update for each parameter.
        """
        for param in self.params:
            if param.requires_grad:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        """
        Resets the gradients of all parameters.
        """
        for param in self.params:
            param.zero_grad()