import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrad import Tensor

a = Tensor([1, 2, 3], requires_grad=True, _name='a')
two = Tensor(2, _name='2')
b = a * two
b._name = 'b'
c = b.sum()
c._name = 'c'

c.backward()

c.graph(show_data=True, show_grad=True)