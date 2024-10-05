import pytest
import numpy as np
import cupy as cp
from minigrad import Tensor

def warm_up_gpu():
    # Perform some dummy operations to warm up the GPU
    dummy_tensor = cp.random.randn(2000, 2000)
    cp.dot(dummy_tensor, dummy_tensor)

@pytest.mark.benchmark(group="Tensor Operations")
@pytest.mark.parametrize("device", ['cpu', 'gpu'])
def test_addition_benchmark(benchmark, device):
    if device == 'gpu':
        warm_up_gpu()  # Warm up GPU before the benchmark

    def add_benchmark(size):
        if device == 'gpu':
            a = Tensor(cp.random.randn(size), device='gpu')
            b = Tensor(cp.random.randn(size), device='gpu')
        else:
            a = Tensor(np.random.randn(size), device='cpu')
            b = Tensor(np.random.randn(size), device='cpu')
        c = a + b
        return c.data

    # Run the benchmark on the add_benchmark function with a tensor size of 10,000
    benchmark.pedantic(add_benchmark, args=[10000], iterations=10, rounds=5)

@pytest.mark.benchmark(group="Tensor Operations")
@pytest.mark.parametrize("device", ['cpu', 'gpu'])
def test_matmul(benchmark, device):
    def matmul_benchmark(size):
        if device == 'gpu':
            a = Tensor(cp.random.randn(size, size), device='gpu')
            b = Tensor(cp.random.randn(size, size), device='gpu')
        else:
            a = Tensor(np.random.randn(size, size), device='cpu')
            b = Tensor(np.random.randn(size, size), device='cpu')
        c = a @ b
        return c.data

    benchmark.pedantic(matmul_benchmark, args=[1000], iterations=10, rounds=5)
