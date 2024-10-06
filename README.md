# minigrad

**minigrad** is a toy implementation of an autograd engine, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) project, with and API similar to that of **PyTorch**. It supports both CPU and GPU operations using `NumPy` and `CuPy`, respectively.

My goal with minigrad is to explore and learn the internals of deep learning frameworks — minigrad is not optimized for speed or efficiency. As of now it provides basic tensor operations, back propagation, neural network layers, optimizers, and loss functions.

## Features

- **Tensor Operations**: Supports basic tensor operations with automatic differentiation.
- **CPU and GPU Support**: Use `NumPy` for CPU operations and `CuPy` for GPU operations.
- **Neural Networks**: Includes simple layers like `Linear` and activations like `ReLU`
- **Optimizers**: Implements basic optimizers like `SGD`.
- **Loss Functions**: Provides loss functions like `MSELoss`.

## Installation

Clone the repository and install minigrad:

```bash
git clone https://github.com/udaysankar01/minigrad.git
cd minigrad
pip install -e .
```

## Run Tests

To run the unit tests and benchmarks for the project

```bash
pytest
```
