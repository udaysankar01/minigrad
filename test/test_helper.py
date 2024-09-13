import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minigrad import Tensor

# Test the shape property
print("Testing the shape property:")
tensor_a = Tensor([[1, 2, 3], [4, 5, 6]])
print("tensor_a data:\n", tensor_a.data)
print("tensor_a shape:", tensor_a.shape)
print()

# Test the zeros method
print("Testing the zeros method:")
zeros_tensor = Tensor.zeros((2, 3))
print("zeros_tensor data:\n", zeros_tensor.data)
print("zeros_tensor shape:", zeros_tensor.shape)
print()

# Test the ones method
print("Testing the ones method:")
ones_tensor = Tensor.ones((2, 3))
print("ones_tensor data:\n", ones_tensor.data)
print("ones_tensor shape:", ones_tensor.shape)
print()

# Test the randn method
print("Testing the randn method:")
randn_tensor = Tensor.randn((2, 3))
print("randn_tensor data:\n", randn_tensor.data)
print("randn_tensor shape:", randn_tensor.shape)
print()

# Test the arange method
print("Testing the arange method:")
arange_tensor = Tensor.arange(0, 10, 2)
print("arange_tensor data:\n", arange_tensor.data)
print("arange_tensor shape:", arange_tensor.shape)
print()