import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrad import Tensor

x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), requires_grad=True, _name='x')

y = x.reshape(3, 3)
y._name = 'y'
print(y)

z = y[:2, :2]
z.backward()
z.graph(show_data=True, show_grad=True)