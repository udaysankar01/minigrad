import numpy as np
from minigrad import Tensor
import minigrad.nn as nn
from minigrad.optim import SGD
from minigrad.loss import MSELoss

class ReLU(nn.Layer):
    def forward(self, x: Tensor)->Tensor:
        return x.relu()

class Simple_NN(nn.Layer):
    """
    A simple feed forward neural network with one hidden layer.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.params = self.fc1.params + self.fc2.params # --> TODO: define automatically (no explcit definition)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x

if __name__ == '__main__':
    np.random.seed(42)

    model = Simple_NN(input_size=2, hidden_size=3, output_size=1)

    optimizer = SGD(params=model.params, lr=0.01)
    criterion = MSELoss()

    x = Tensor(np.random.rand(1, 2), requires_grad=False)
    y_true = Tensor([[0.5]], requires_grad=False)

    # training loop
    epoch = 100
    for epoch in range(epoch):
        
        y_pred = model.forward(x)
        loss = criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        loss.graph(show_data=True, show_grad=True)
        optimizer.step() 

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data}")
    
    final_output = model.forward(x)
    print("Final output after training: ", final_output)