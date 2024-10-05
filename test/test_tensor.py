import unittest
import cupy as cp
import numpy as np
from minigrad import Tensor

class BaseTestTensor(unittest.TestCase):
    
    def run_test_on_devices(self, test_func):
        """
        Helper function to loop through available devices ('cpu' and 'gpu') and
        run the given test function
        """
        for device in ['cpu', 'gpu']:
            with self.subTest(device=device):
                test_func(device)

    def assert_tensor_properties(self, tensor, data, device, grad=None, requires_grad=True):
        self.assertEqual(tensor.device, device)
        cp.testing.assert_array_almost_equal(tensor.data, data)
        if device == "cpu":
            self.assertTrue(isinstance(tensor.data, np.ndarray))
        else:
            self.assertTrue(isinstance(tensor.data, cp.ndarray))

        if grad is not None:
            cp.testing.assert_array_almost_equal(tensor.grad, grad)
        self.assertEqual(tensor.requires_grad, requires_grad)

class TestTensorCreation(BaseTestTensor):

    def _run_tensor_creation_test(self, device):
        t = Tensor([1, 2, 3], device=device)                                                # from list
        self.assert_tensor_properties(t, [1, 2, 3], device)                           
        t = Tensor(np.array([4, 5, 6]), requires_grad=False, device=device)                 # from numpy array
        self.assert_tensor_properties(t, [4, 5, 6], device, requires_grad=False)
        t = Tensor(cp.array([7, 8, 9]), requires_grad=False, device=device)                 # from cupy array
        self.assert_tensor_properties(t, [7, 8, 9], device, requires_grad=False)
    
    def _run_tensor_creation_from_scalar_test(self, device):
        t = Tensor(10.0, device=device)
        self.assert_tensor_properties(t, 10.0, device)

    def test_tensor_creation_cpu(self):
        self._run_tensor_creation_test('cpu')

    def test_tensor_creation_gpu(self):
        self._run_tensor_creation_test('gpu')
    
    def test_tensor_creation_from_scalar_cpu(self):
        self._run_tensor_creation_from_scalar_test('cpu')
    
    def test_tensor_creation_from_scalar_gpu(self):
        self._run_tensor_creation_from_scalar_test('gpu')

class TestTensorTransfer(BaseTestTensor):

    def test_tensor_transfer(self):
        # create a tensor on CPU, transfer to GPU and then back to CPU
        data_cpu = np.array([1, 2, 3], dtype=np.float32)
        t_cpu = Tensor(data_cpu, device='cpu')
        self.assert_tensor_properties(t_cpu, data_cpu, 'cpu', requires_grad=True)

        t_gpu = t_cpu.to('gpu')
        self.assert_tensor_properties(t_gpu, data_cpu, 'gpu', requires_grad=True)

        t_cpu_back = t_gpu.to('cpu')
        self.assert_tensor_properties(t_cpu_back, data_cpu, 'cpu', requires_grad=True)

class TestElementwiseTensorOps(BaseTestTensor):

    def _run_add_test(self, device):
        # TODO: Add broadcasting test
        # TODO: Add a + a test
        # Tensor + Tensor
        a = Tensor([1, 2, 3], device=device)
        b = Tensor([4, 5, 6], device=device)
        c = a + b
        c.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(c, [5, 7, 9], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [1, 2, 1])
        self.assert_tensor_properties(b, [4, 5, 6], device, [1, 2, 1])

        # a + a
        a.zero_grad()
        d = a + a
        d.backward(Tensor([1, 1, 1], device=device))
        self.assert_tensor_properties(d, [2, 4, 6], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [2, 2, 2])

    def _run_add_scalar_test(self, device):
        # Tensor + Scalar
        a = Tensor([1, 2, 3], device=device)
        b = a + 5
        b.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(b, [6, 7, 8], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [1, 2, 1])

        # Scalar + Tensor
        a.zero_grad()
        self.assert_tensor_properties(a, [1, 2, 3], device, None)
        c = 6 + a
        c.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(c, [7, 8, 9], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [1, 2, 1])

    def _run_sub_test(self, device):
        # TODO: Add broadcasting test
        # TODO: Add a - a test
        # Tensor1 - Tensor2
        a = Tensor([5, 6, 7], device=device)
        b = Tensor([1, 2, 3], device=device)
        c = a - b
        c.backward(Tensor([1, 2, 1]))

        self.assert_tensor_properties(c, [4, 4, 4], device)
        self.assert_tensor_properties(a, [5, 6, 7], device, [1, 2, 1])
        self.assert_tensor_properties(b, [1, 2, 3], device, [-1, -2, -1])

        # Tensor2 - Tensor1
        a.zero_grad()
        b.zero_grad()
        d = b - a
        d.backward(Tensor([1, 2, 1]))

        self.assert_tensor_properties(d, [-4, -4, -4], device)
        self.assert_tensor_properties(a, [5, 6, 7], device, [-1, -2, -1])
        self.assert_tensor_properties(b, [1, 2, 3], device, [1, 2, 1])

    def _run_sub_scalar_test(self, device):
        # Tensor - scalar
        a = Tensor([5, 6, 7], device=device)
        b = a - 2
        b.backward(Tensor([1, 2, 1]))
        self.assert_tensor_properties(b, [3, 4, 5], device)
        self.assert_tensor_properties(a, [5, 6, 7], device, [1, 2, 1])

        # Scalar - Tensor
        a.zero_grad()
        self.assert_tensor_properties(a, [5, 6, 7], device, None)
        c = 10 - a
        c.backward(Tensor([1, 2, 1]))
        self.assert_tensor_properties(c, [5, 4, 3], device)
        self.assert_tensor_properties(a, [5, 6, 7], device, [-1, -2, -1])

    def _run_mul_test(self, device):
        # TODO: Add broadcasting test
        # TODO: Add a * a test
        # Tensor1 * Tensor2
        a = Tensor([1, 2, 3], device=device)
        b = Tensor([4, 5, 6], device=device)
        c = a * b
        c.backward(Tensor([1, 2, 1], device=device))

        self.assert_tensor_properties(c, [4, 10, 18], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [4, 10, 6])
        self.assert_tensor_properties(b, [4, 5, 6], device, [1, 4, 3])

    def _run_mul_scalar_test(self, device):
        # Tensor * Scalar
        a = Tensor([1, 2, 3], device=device)
        b = a * 5
        b.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(b, [5, 10, 15], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [5, 10, 5])

        # Scalar * Tensor
        a.zero_grad()
        self.assert_tensor_properties(a, [1, 2, 3], device, None)
        c = 6 * a
        c.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(c, [6, 12, 18], device)
        self.assert_tensor_properties(a, [1, 2, 3], device, [6, 12, 6])

    def _run_div_test(self, device):
        # TODO: Add broadcasting test
        # TODO: Add a / a test
        # Tensor1 / Tensor2
        a = Tensor([10, 15, 20], device=device)
        b = Tensor([2, 3, 4], device=device)
        c = a / b
        c.backward(Tensor([1, 2, 1], device=device))

        self.assert_tensor_properties(c, [5, 5, 5], device)
        self.assert_tensor_properties(a, [10, 15, 20], device, [0.5, 2/3, 0.25])
        self.assert_tensor_properties(b, [2, 3, 4], device, [-2.5, -10/3, -1.25])

        # Tensor2 / Tensor1
        a.zero_grad()
        b.zero_grad()
        d = b / a
        d.backward(Tensor([1, 2, 1], device=device))

        self.assert_tensor_properties(d, [0.2, 0.2, 0.2], device)
        self.assert_tensor_properties(a, [10, 15, 20], device, [-1/50, -2/75, -0.01])
        self.assert_tensor_properties(b, [2, 3, 4], device, [0.1, 2/15, 1/20])
    
    def _run_div_scalar_test(self, device):
        # Tensor / scalar
        a = Tensor([4, 9, 16], device=device)
        b = a / 2
        b.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(b, [2, 4.5, 8], device)
        self.assert_tensor_properties(a, [4, 9, 16], device, [0.5, 1, 0.5])

        # scalar / Tensor
        a.zero_grad()
        self.assert_tensor_properties(a, [4, 9, 16], device, None)
        c = 32 / a
        c.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(c, [8, 32/9, 2], device)
        self.assert_tensor_properties(a, [4, 9, 16], device, [-2, -64/81, -1/8])

    def test_addition(self):
        self.run_test_on_devices(self._run_add_test)
    
    def test_addition_scalar(self):
        self.run_test_on_devices(self._run_add_scalar_test)
    
    def test_subtraction(self):
        self.run_test_on_devices(self._run_sub_test)
    
    def test_subtraction_scalar(self):
        self.run_test_on_devices(self._run_sub_scalar_test)

    def test_multiplication(self):
        self.run_test_on_devices(self._run_mul_test)

    def test_multiplication_scalar(self):
        self.run_test_on_devices(self._run_mul_scalar_test)

    def test_division(self):
        self.run_test_on_devices(self._run_div_test)
    
    def test_division_scalar(self):
        self.run_test_on_devices(self._run_div_scalar_test)

class TestMatrixTensorOps(BaseTestTensor):

    def _run_matmul_test(self, device):
        # Tensor1 @ Tensor2
        a = Tensor([[1, 2], [3, 4]], device=device)
        b = Tensor([[5, 6], [7, 8]], device=device)
        c = a @ b
        c.backward(Tensor([[1, 1], [1, 1]], device=device))

        self.assert_tensor_properties(c, [[19, 22], [43, 50]], device)
        self.assert_tensor_properties(a, [[1, 2], [3, 4]], device, [[11, 15], [11, 15]])
        self.assert_tensor_properties(b, [[5, 6], [7, 8]], device, [[4, 4], [6, 6]])

        # Tensor2 @ Tensor1
        a.zero_grad()
        b.zero_grad()
        d = b @ a
        d.backward(Tensor([[1, 1], [1, 1]], device=device))

        self.assert_tensor_properties(d, [[23, 34], [31, 46]], device)
        self.assert_tensor_properties(a, [[1, 2], [3, 4]], device, [[12, 12], [14, 14]])
        self.assert_tensor_properties(b, [[5, 6], [7, 8]], device, [[3, 7], [3, 7]])
    
    def test_matmul(self):
        self.run_test_on_devices(self._run_matmul_test)

class TestUnaryTensorOps(BaseTestTensor):

    def _run_negation_test(self, device):
        a = Tensor([1, -2, 3], device=device)
        b = -a
        b.backward(Tensor([1, 2, 1], device=device))
        self.assert_tensor_properties(b, [-1, 2, -3], device)
        self.assert_tensor_properties(a, [1, -2, 3], device, [-1, -2, -1])

    def _run_sum_test(self, device):
        # sum of all elements
        a_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        a = Tensor(a_data, device=device)
        b = a.sum()
        b.backward(Tensor(2, device=device))
        self.assert_tensor_properties(b, 36, device)
        self.assert_tensor_properties(a, a_data, device, 2 * np.ones_like(a_data))

        # sum along axis 0
        a.zero_grad()
        c = a.sum(axis=0)
        c.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(c, [[6, 8], [10, 12]], device)
        self.assert_tensor_properties(a, a_data, device, [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

        # sum along axis 1
        a.zero_grad()
        d = a.sum(axis=1)
        d.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(d, [[4, 6], [12, 14]], device)
        self.assert_tensor_properties(a, a_data, device, [[[1, 2], [1, 2]], [[3, 4], [3, 4]]])

        # sum along axis 2
        a.zero_grad()
        e = a.sum(axis=2)
        e.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(e, [[3, 7], [11, 15]], device)
        self.assert_tensor_properties(a, a_data, device, [[[1, 1], [2, 2]], [[3, 3], [4, 4]]])

    def _run_mean_test(self, device):
        # sum of all elements
        a_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        a = Tensor(a_data, device=device)
        b = a.mean()
        b.backward(Tensor(2, device=device))
        self.assert_tensor_properties(b, 4.5, device)
        self.assert_tensor_properties(a, a_data, device, 0.25 * np.ones_like(a_data))

        # sum along axis 0
        a.zero_grad()
        c = a.mean(axis=0)
        c.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(c, [[3, 4], [5, 6]], device)
        self.assert_tensor_properties(a, a_data, device, [[[0.5, 1], [1.5, 2]], [[0.5, 1], [1.5, 2]]])

        # sum along axis 1
        a.zero_grad()
        d = a.mean(axis=1)
        d.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(d, [[2, 3], [6, 7]], device)
        self.assert_tensor_properties(a, a_data, device, [[[0.5, 1], [0.5, 1]], [[1.5, 2], [1.5, 2]]])

        # sum along axis 2
        a.zero_grad()
        e = a.mean(axis=2)
        e.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(e, [[1.5, 3.5], [5.5, 7.5]], device)
        self.assert_tensor_properties(a, a_data, device, [[[0.5, 0.5], [1, 1]], [[1.5, 1.5], [2, 2]]])

    def _run_relu_test(self, device):
        a = Tensor([-1, 0, 1], device=device)
        b = a.relu()
        b.backward(Tensor([1, 1, 1], device=device))
        self.assert_tensor_properties(b, [0, 0, 1], device)
        self.assert_tensor_properties(a, [-1, 0, 1], device, [0, 0, 1])

    def test_negation(self):
        self.run_test_on_devices(self._run_negation_test)
    
    def test_sum(self):
        self.run_test_on_devices(self._run_sum_test)
    
    def test_mean(self):
        self.run_test_on_devices(self._run_mean_test)

    def test_relu(self):
        self.run_test_on_devices(self._run_relu_test)

class TestMatrixManipTensorOps(BaseTestTensor):
    
    def _run_transpose_test(self, device):
        a = Tensor([[1, 2, 3], [4, 5, 6]], device=device)
        b = a.T()
        b.backward(Tensor([[1, 2], [3, 4], [5, 6]], device=device))
        self.assert_tensor_properties(b, [[1, 4], [2, 5], [3, 6]], device)
        self.assert_tensor_properties(a, [[1, 2, 3], [4, 5, 6]], device, [[1, 3, 5], [2, 4, 6]])

    def _run_reshape_test(self, device):
        # reshape from (4,) to (2, 2) and back to (4,)
        a = Tensor([1, 2, 3, 4], device=device)
        b = a.reshape(2, 2)
        b.backward(Tensor([[1, 2], [3, 4]], device=device))
        self.assert_tensor_properties(b, [[1, 2], [3, 4]], device)
        self.assert_tensor_properties(a, [1, 2, 3, 4], device, [1, 2, 3, 4])

        a.zero_grad()
        b.zero_grad()
        self.assert_tensor_properties(a, [1, 2, 3, 4], device, None)
        self.assert_tensor_properties(b, [[1, 2], [3, 4]], device, None)
        a_back = b.reshape(4)
        a_back.backward(Tensor([[1, 2, 3, 4]], device=device))
        self.assert_tensor_properties(a_back, [1, 2, 3, 4], device)
        self.assert_tensor_properties(b, [[1, 2], [3, 4]], device, [[1, 2], [3, 4]])
        self.assert_tensor_properties(a, [1, 2, 3, 4], device, [1, 2, 3, 4])

    def _run_slicing_test(self, device):
        a = Tensor([[1, 2, 3], [4, 5, 6]], device=device)
        c = a[1, :]
        c.backward(Tensor([1, 2, 3], device=device))
        self.assert_tensor_properties(c, [4, 5, 6], device)
        self.assert_tensor_properties(a, [[1, 2, 3], [4, 5, 6]], device, [[0, 0, 0], [1, 2, 3]])

    def test_transpose(self):
        self.run_test_on_devices(self._run_transpose_test)
    
    def test_reshape(self):
        self.run_test_on_devices(self._run_reshape_test)
    
    def test_slicing(self):
        self.run_test_on_devices(self._run_slicing_test)

class TestTensorUtilityOps(BaseTestTensor):

    def _run_zero_grad_test(self, device):
        a = Tensor([1, 2, 3], device=device)
        b = a * 2
        c = b.sum()
        c.backward()
        self.assert_tensor_properties(c, 12, device)
        self.assert_tensor_properties(b, [2, 4, 6], device, [1, 1, 1])
        self.assert_tensor_properties(a, [1, 2, 3], device, [2, 2, 2])
        
        c.zero_grad()
        self.assert_tensor_properties(c, 12, device)
        self.assert_tensor_properties(b, [2, 4, 6], device)
        self.assert_tensor_properties(a, [1, 2, 3], device)
    
    def _run_helper_methods_test(self, device):
        # zeros
        a = Tensor.zeros((2, 2), device=device)
        self.assert_tensor_properties(a, [[0, 0], [0, 0]], device, requires_grad=True)

        # ones
        a = Tensor.ones((2, 2), device=device)
        self.assert_tensor_properties(a, [[1, 1], [1, 1]], device, requires_grad=True)

        # randn -- TODO: improve test (SEED?)
        a = Tensor.randn((2, 2), device=device)
        self.assert_tensor_properties(a, a.data, device, requires_grad=True)

        # arange
        a = Tensor.arange(0, 5, device=device)
        self.assert_tensor_properties(a, [0, 1, 2, 3, 4], device, requires_grad=True)

    def _run_shape_test(self, device):
        a = Tensor([[1, 2, 3], [4, 5, 6]], device=device)
        shape = a.shape
        self.assertEqual(shape, (2, 3))

    def _run_chain_rule_test(self, device):
        a = Tensor(2.0, device=device)
        b = a + 2
        c = a * b
        c.backward()
        self.assert_tensor_properties(c, 8, device)
        self.assert_tensor_properties(a, 2, device, 6)
        self.assert_tensor_properties(b, 4, device, 2)

    def _run_non_scalar_backward_test(self, device):
        a = Tensor([1, 2, 3], device=device)
        b = Tensor([4, 5, 6], device=device)
        c = a + b # non-scalar output

        # calling backward for non-scalar output without gradient raises an error
        with self.assertRaises(ValueError) as context:
            c.backward()
        self.assertEqual(str(context.exception), "grad must be provided for non-scalar outputs.")
    
    def _run_unbroadcast_grad_test(self, device):
        # testing unbroadcasting of gradients when adding two tensors of different shapes. TODO: remove
        a = Tensor([[1], [2], [3]], device=device)
        b = Tensor([4, 5, 6], device=device)
        c = a + b # non-scalar output
        c_sum = c.sum()
        c_sum.backward()
        self.assert_tensor_properties(c_sum, 63, device)
        self.assert_tensor_properties(c, [[5, 6, 7], [6, 7, 8], [7, 8, 9]], device)
        self.assert_tensor_properties(b, [4, 5, 6], device, [3, 3, 3])
        self.assert_tensor_properties(a, [[1], [2], [3]], device, [[3], [3], [3]])

    def _run_custom_grad_test(self, device):
        a = Tensor([1, -2, 3], device=device)
        b = a.relu()
        grad = Tensor([0.1, 0.2, 0.3], device=device)
        b.backward(grad)
        self.assert_tensor_properties(b, [1, 0, 3], device)
        self.assert_tensor_properties(a, [1, -2, 3], device, [0.1, 0, 0.3])

    def test_zero_grad(self):
        self.run_test_on_devices(self._run_zero_grad_test)
    
    def test_hepler_methods(self):
        self.run_test_on_devices(self._run_helper_methods_test)
    
    def test_shape_property(self):
        self.run_test_on_devices(self._run_shape_test)
    
    def test_chain_rule(self):
        self.run_test_on_devices(self._run_chain_rule_test)

    def test_non_scalar_backward(self):
        self.run_test_on_devices(self._run_non_scalar_backward_test)
    
    def test_unbroadcast_grad(self):
        self.run_test_on_devices(self._run_unbroadcast_grad_test)

    def test_backward_pass_custom_grad(self):
        self.run_test_on_devices(self._run_custom_grad_test)

class TestGraphVisualization(BaseTestTensor):

    def _run_graph_visualization_test(self, device):
        # test whether the graph method runs without any error (visual verification not possible here)
        a = Tensor([1.0, 2.0], device=device, _name='a')
        b = Tensor([3.0, 4.0], device=device, _name='b')
        c = a * b
        d = c.sum()
        d.backward()
        try:
            d.graph(show_data=True, show_grad=True)
        except Exception as e:
            self.fail(f"Graph method raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
