import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrad import Tensor

a = Tensor([1, 2, 3], requires_grad=True)
b = a * 2

b.backward()



b.graph(show_data=True, show_grad=True)