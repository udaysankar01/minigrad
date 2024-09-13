import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrad import Tensor

x = Tensor(np.array([1, 2, 3, 4]), requires_grad=True)
y = x.reshape(2, 2)
y.backward()
print(x.grad)