# minigrad

A toy implementation of autograd engine

## Project Structure

```bash
minigrad/
├── __init__.py               # Initialize the module
├── tensor.py                 # Core Tensor
├── autograd.py               # Backpropagation logic
├── nn.py                     # Simple neural network layers and activations
├── optim.py                  # Optimizers (SGD, Adam, etc.)
└── loss.py                   # Loss functions (MSE, Cross-entropy, etc.)
```

## Run Tests

```bash
python -m unittest discover -s test
```
