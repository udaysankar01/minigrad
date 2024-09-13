import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrad import Tensor
from minigrad.nn import Linear

# define input tensor (batch_size=2, in_features=3)
input_data = np.random.randn(2, 3)
input_tensor = Tensor(input_data)

# initialize the Linear layer (in_features=3, out_features=2)
linear_layer = Linear(in_features=3, out_features=2)

# perform a forward pass through the linear layer
output_tensor = linear_layer.forward(input_tensor)

print("Input tensor:\n")
print(input_tensor)

print("Output tensor:\n")
print(output_tensor)