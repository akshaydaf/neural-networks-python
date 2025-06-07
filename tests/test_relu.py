import unittest
import numpy as np
from utilities.relu import relu_forward, relu_backward


class TestReLU(unittest.TestCase):
    def test_relu_forward(self):
        """Test that relu_forward correctly applies the ReLU function.
        
        :return: None
        """
        x = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 0.5]])
        expected = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 0.5]])
        out = relu_forward(x)
        np.testing.assert_array_almost_equal(out, expected)

    def test_relu_backward(self):
        """Test that relu_backward correctly computes the derivative of ReLU.
        
        :return: None
        """
        x = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 0.5]])
        _ = relu_forward(x)
        expected_dx = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        dx = relu_backward(x)

        np.testing.assert_array_almost_equal(dx, expected_dx)
