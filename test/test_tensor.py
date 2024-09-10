import unittest
import numpy as np
import torch
from minigrad.tensor import Tensor

class TestTensor(unittest.TestCase):

    def test_tensor_creation(self):
        # test creation from NumPy array in minigrad
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        minigrad_tensor = Tensor(data, requires_grad=False)
        pytorch_tensor = torch.Tensor(data)

        # compare data
        np.testing.assert_allclose(minigrad_tensor.data, pytorch_tensor.detach().numpy(), atol=1e-5)
        # compare requires_grad flag
        self.assertEqual(minigrad_tensor.requires_grad, pytorch_tensor.requires_grad)
        # # gradient initially None in both cases
        self.assertIsNone(minigrad_tensor.grad, pytorch_tensor.grad)

if __name__ == '__main__':
    unittest.main()