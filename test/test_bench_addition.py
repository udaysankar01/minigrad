import pytest
import numpy as np
import cupy as cp
from minigrad import Tensor

@pytest.mark.benchmark(group="addition")
@pytest.mark.parametrize("device", ['cpu', 'gpu'])
def test_addition_benchmark(benchmark, device):
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
