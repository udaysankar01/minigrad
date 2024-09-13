import numpy as np
from minigrad import Tensor

# Set seed for reproducibility
np.random.seed(0)

# Define dimensions
batch_size = 2
input_dim = 3
output_dim = 4

# Create input tensor x
x_data = np.random.randn(batch_size, input_dim)
x = Tensor(x_data, requires_grad=True, _name='x')

# Create weights tensor W
W_data = np.random.randn(input_dim, output_dim)
W = Tensor(W_data, requires_grad=True, _name='W') # (3, 4)

# Create bias tensor b
b_data = np.random.randn(1, output_dim)
b = Tensor(b_data, requires_grad=True, _name='b') # (1, 4)

# Forward pass: y = x @ W + b
product = x @ W
product._name = 'x @ W'
y = product + b  # y is (2, 4)
y._name = 'y'

# Compute loss
loss = y.sum()
loss._name = 'loss'

# Backward pass
loss.backward()

# Print gradients
print("Gradients computed by Tensor class:")
print("dloss/dW:\n", W.grad)
print("dloss/dy:\n", y.grad)
print("dloss/db:\n", b.grad)
print("dloss/dx:\n", x.grad)

# Compute expected gradients using NumPy
dloss_dy = np.ones_like(y.data)
dloss_dW = x.data.T @ dloss_dy
dloss_db = dloss_dy.sum(axis=0, keepdims=True)
dloss_dx = dloss_dy @ W.data.T

# Compare
print("\nExpected gradients computed using NumPy:")
print("dloss/dW:\n", dloss_dW)
print("dloss/db:\n", dloss_db)
print("dloss/dx:\n", dloss_dx)

# Visualize the computational graph
dot = loss.graph("example")

# Verify gradients
assert np.allclose(W.grad, dloss_dW), "Gradient w.r.t W does not match"
assert np.allclose(b.grad, dloss_db), "Gradient w.r.t b does not match"
assert np.allclose(x.grad, dloss_dx), "Gradient w.r.t x does not match"

print("\nAll gradients match expected values.")