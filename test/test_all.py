import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrad import Tensor, SGD, MSELoss
import minigrad.nn as nn

# define a simple neural network
class SimpleNN:
    def __init__(self):
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x)
        return x
    
    def parameters(self):
        return self.layer1.params + self.layer2.params

# Create random data
np.random.seed(42)
X = Tensor(np.random.randn(100, 2), requires_grad=False)  # 100 samples, 2 features
y = Tensor(np.random.randn(100, 1), requires_grad=False)  # 100 samples, 1 target

# Initialize the neural network, loss function, and optimizer
model = SimpleNN()
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# training loop
epochs = 100
for epoch in range(epochs):
    pred = model.forward(X)

    # compute loss
    loss = loss_fn(pred, y)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update parameters
    optimizer.step()

    # print loss every 10 epochs
    if (epoch % 10) == 0:
        print(f"Epoch: {epoch}, Loss: {loss.data: .3f}")

test_input = Tensor(np.random.randn(10, 2), requires_grad=False)
test_output = model.forward(test_input)
print("Test output:", test_output.data)