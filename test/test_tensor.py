import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import minigrad as mg
from minigrad import Tensor

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import cupy as cp
from minigrad import Tensor

class TestTensor(unittest.TestCase):

    def test_tensor_creation_cpu(self):
        # Test creating cpu tensor from a list
        t = Tensor([1, 2, 3], device='cpu')
        self.assertTrue(isinstance(t.data, np.ndarray))
        np.testing.assert_array_equal(t.data, np.array([1, 2, 3]))
        self.assertTrue(t.requires_grad)
        self.assertEqual(t.device, 'cpu')

        # Test creating cpu tensor from a numpy array
        data = np.array([4, 5, 6])
        t = Tensor(data, requires_grad=False, device='cpu')
        cp.testing.assert_array_equal(t.data, data)
        self.assertFalse(t.requires_grad)
        self.assertEqual(t.device, 'cpu')

        # Test creating cpu tensor from a cupy array
        data = cp.array([4, 5, 6])
        t = Tensor(data, requires_grad=False, device='cpu')
        cp.testing.assert_array_equal(t.data, data)
        self.assertFalse(t.requires_grad)
        self.assertEqual(t.device, 'cpu')

        # Test creating cpu tensor from a scalar
        t = Tensor(10.0, device='cpu')
        self.assertEqual(t.data, 10.0)
        self.assertTrue(t.requires_grad)
        self.assertEqual(t.device, 'cpu')

    def test_tensor_creation_gpu(self):
        # Test creating gpu tensor from a list
        t = Tensor([1, 2, 3], device='gpu')
        self.assertEqual(t.device, 'gpu')
        self.assertTrue(isinstance(t.data, cp.ndarray))
        cp.testing.assert_array_equal(t.data, cp.array([1, 2, 3]))
        self.assertEqual(t.device, 'gpu')
        self.assertTrue(t.requires_grad)

        # Test creating gpu tensor from a numpy array
        data = np.array([4, 5, 6])
        t = Tensor(data, requires_grad=False, device='gpu')
        cp.testing.assert_array_equal(t.data, data)
        self.assertFalse(t.requires_grad)
        self.assertEqual(t.device, 'gpu')

        # Test creating gpu tensor from a cupy array
        data = cp.array([4, 5, 6])
        t = Tensor(data, requires_grad=False, device='gpu')
        cp.testing.assert_array_equal(t.data, data)
        self.assertFalse(t.requires_grad)
        self.assertEqual(t.device, 'gpu')

        # Test creating gpu tensor from a scalar
        t = Tensor(10.0, device='gpu')
        self.assertEqual(t.data, 10.0)
        self.assertTrue(t.requires_grad)
        self.assertEqual(t.device, 'gpu')


    def test_addition(self):
        # Tensor + Tensor
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        np.testing.assert_array_equal(c.data, [5, 7, 9])

        # Backward pass
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [1, 1, 1])
        np.testing.assert_array_equal(b.grad, [1, 1, 1])

        # Tensor + scalar
        a.zero_grad()
        c = a + 5
        np.testing.assert_array_equal(c.data, [6, 7, 8])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [1, 1, 1])

        # Scalar + Tensor
        a.zero_grad()
        c = 5 + a
        np.testing.assert_array_equal(c.data, [6, 7, 8])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [1, 1, 1])

    def test_subtraction(self):
        # Tensor - Tensor
        a = Tensor([5, 6, 7], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a - b
        np.testing.assert_array_equal(c.data, [4, 4, 4])

        # Backward pass
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [1, 1, 1])
        np.testing.assert_array_equal(b.grad, [-1, -1, -1])

        # Tensor - scalar
        a.zero_grad()
        c = a - 2
        np.testing.assert_array_equal(c.data, [3, 4, 5])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [1, 1, 1])

        # Scalar - Tensor
        a.zero_grad()
        c = 10 - a
        np.testing.assert_array_equal(c.data, [5, 4, 3])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [-1, -1, -1])

    def test_multiplication(self):
        # Tensor * Tensor
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a * b
        np.testing.assert_array_equal(c.data, [4, 10, 18])

        # Backward pass
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, b.data)
        np.testing.assert_array_equal(b.grad, a.data)

        # Tensor * scalar
        a.zero_grad()
        c = a * 2
        np.testing.assert_array_equal(c.data, [2, 4, 6])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [2, 2, 2])

        # Scalar * Tensor
        a.zero_grad()
        c = 3 * a
        np.testing.assert_array_equal(c.data, [3, 6, 9])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [3, 3, 3])

    def test_division(self):
        # Tensor / Tensor
        a = Tensor([4.0, 9.0, 16.0], requires_grad=True)
        b = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        c = a / b
        np.testing.assert_array_almost_equal(c.data, [2.0, 3.0, 4.0])

        # Backward pass
        c.backward(Tensor([1.0, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(a.grad, [1/2.0, 1/3.0, 1/4.0])
        np.testing.assert_array_almost_equal(
            b.grad, [-4.0/(2.0**2), -9.0/(3.0**2), -16.0/(4.0**2)])

        # Tensor / scalar
        a.zero_grad()
        c = a / 2
        np.testing.assert_array_equal(c.data, [2.0, 4.5, 8.0])
        c.backward(Tensor([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(a.grad, [0.5, 0.5, 0.5])

        # Scalar / Tensor
        a.zero_grad()
        c = 32 / a
        np.testing.assert_array_almost_equal(
            c.data, [8.0, 32.0/9.0, 2.0])
        c.backward(Tensor([1.0, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(
            a.grad, [-32.0/(4.0**2), -32.0/(9.0**2), -32.0/(16.0**2)])

    def test_negation(self):
        a = Tensor([1, -2, 3], requires_grad=True)
        b = -a
        np.testing.assert_array_equal(b.data, [-1, 2, -3])
        b.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [-1, -1, -1])

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b
        np.testing.assert_array_equal(c.data, [[19, 22], [43, 50]])

        # Backward pass
        c.backward(Tensor([[1, 1], [1, 1]]))
        np.testing.assert_array_equal(a.grad, np.array([[11, 15], [11, 15]]))
        np.testing.assert_array_equal(b.grad, np.array([[4, 4], [6, 6]]))

    def test_sum(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        c = a.sum()
        self.assertEqual(c.data, 10)
        c.backward()
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data))

        # Sum along axis
        a.zero_grad()
        c = a.sum(axis=0)
        np.testing.assert_array_equal(c.data, [4, 6])
        c.backward(Tensor([1, 1]))
        np.testing.assert_array_equal(a.grad, [[1, 1], [1, 1]])

    def test_mean(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = a.mean()
        self.assertEqual(c.data, 2.5)
        c.backward()
        np.testing.assert_array_equal(a.grad, np.ones_like(a.data) * 0.25)

        # Mean along axis
        a.zero_grad()
        c = a.mean(axis=0)
        np.testing.assert_array_equal(c.data, [2.0, 3.0])
        c.backward(Tensor([1.0, 1.0]))
        np.testing.assert_array_equal(a.grad, [[0.5, 0.5], [0.5, 0.5]])

    def test_relu(self):
        a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = a.relu()
        np.testing.assert_array_equal(c.data, [0.0, 0.0, 1.0])
        c.backward(Tensor([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(a.grad, [0.0, 0.0, 1.0])

    def test_transpose(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        c = a.T()
        np.testing.assert_array_equal(c.data, [[1, 4], [2, 5], [3, 6]])
        c.backward(Tensor([[1, 1], [1, 1], [1, 1]]))
        np.testing.assert_array_equal(a.grad, [[1, 1, 1], [1, 1, 1]])

    def test_reshape(self):
        a = Tensor([1, 2, 3, 4], requires_grad=True)
        c = a.reshape(2, 2)
        np.testing.assert_array_equal(c.data, [[1, 2], [3, 4]])
        c.backward(Tensor([[1, 1], [1, 1]]))
        np.testing.assert_array_equal(a.grad, [1, 1, 1, 1])

    def test_slicing(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        c = a[1, :]
        np.testing.assert_array_equal(c.data, [4, 5, 6])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [[0, 0, 0], [1, 1, 1]])

    def test_zero_grad(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a * 2
        c = b.sum()
        c.backward()
        np.testing.assert_array_equal(a.grad, [2, 2, 2])
        a.zero_grad()
        self.assertIsNone(a.grad)

    def test_helper_methods(self):
        # zeros
        a = Tensor.zeros((2, 2))
        np.testing.assert_array_equal(a.data, np.zeros((2, 2)))
        self.assertFalse(a.requires_grad)

        # ones
        a = Tensor.ones((2, 2), requires_grad=True)
        np.testing.assert_array_equal(a.data, np.ones((2, 2)))
        self.assertTrue(a.requires_grad)

        # randn
        a = Tensor.randn((2, 2))
        self.assertEqual(a.data.shape, (2, 2))

        # arange
        a = Tensor.arange(0, 5)
        np.testing.assert_array_equal(a.data, [0, 1, 2, 3, 4])

    def test_shape_property(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.shape, (2, 3))

    def test_broadcasting(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor(2, requires_grad=True)
        c = a * b
        np.testing.assert_array_equal(c.data, [2, 4, 6])
        c.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [2, 2, 2])
        self.assertEqual(b.grad, a.data.sum())

    def test_chain_rule(self):
        # Compute gradient of c = a * b, where a = b + 2
        b = Tensor(2.0, requires_grad=True)
        a = b + 2
        c = a * b
        c.backward()
        # Expected gradients:
        # dc/db = da/db * dc/da + dc/db
        # da/db = 1
        # dc/da = b
        # dc/db = a + b
        expected_grad_b = a.data + b.data  # Should be 4 + 2 = 6
        self.assertEqual(b.grad, expected_grad_b)

    def test_non_scalar_backward(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = 2 * a
        b.backward(Tensor([1, 1, 1]))
        np.testing.assert_array_equal(a.grad, [2, 2, 2])

    def test_unbroadcast_grad(self):
        # Test unbroadcasting of gradients
        a = Tensor([[1], [2], [3]], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b  # Broadcasting happens here
        c_sum = c.sum()
        c_sum.backward()
        np.testing.assert_array_equal(a.grad, np.array([[3], [3], [3]]))
        np.testing.assert_array_equal(b.grad, np.array([3, 3, 3]))

    def test_backward_pass_with_non_scalar_output(self):
        # Ensure that backward pass works with non-scalar outputs when grad is provided
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = a * 2
        grad = Tensor([[1, 1], [1, 1]])
        b.backward(grad)
        np.testing.assert_array_equal(a.grad, np.array([[2, 2], [2, 2]]))

    def test_graph_method(self):
        # Test the graph method runs without error (visual verification not possible here)
        a = Tensor([1.0, 2.0], requires_grad=True, _name='a')
        b = Tensor([3.0, 4.0], requires_grad=True, _name='b')
        c = a * b
        d = c.sum()
        d.backward()
        try:
            d.graph(show_data=True, show_grad=True)
        except Exception as e:
            self.fail(f"Graph method raised an exception: {e}")

    def test_backward_pass_with_custom_grad(self):
        # Test backward pass when a custom gradient is provided
        a = Tensor([1.0, -2.0, 3.0], requires_grad=True)
        b = a.relu()
        grad = Tensor([0.1, 0.2, 0.3])
        b.backward(grad)
        np.testing.assert_array_equal(a.grad, np.array([0.1, 0.0, 0.3], dtype=np.float32))

if __name__ == '__main__':
    unittest.main()
