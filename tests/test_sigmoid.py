import unittest
import numpy as np
import math
from utilities.sigmoid import (
    sigmoid_forward,
    sigmoid_backward,
)


class TestSigmoidFunctions(unittest.TestCase):
    def test_sigmoid_forward_scalar(self):
        """Test that sigmoid_forward correctly applies the Sigmoid function for a scalar.

        :return: None
        """
        x = 0
        expected = 0.5
        result = sigmoid_forward(x)
        self.assertTrue(math.isclose(result, expected, rel_tol=1e-6))

    def test_sigmoid_forward_array(self):
        """Test that sigmoid_forward correctly applies the Sigmoid function for an array.

        :return: None
        """
        xs = np.array([-1.0, 0.0, 1.0])
        expected = np.array([1 / (1 + math.exp(1)), 0.5, 1 / (1 + math.exp(-1))])
        result = np.array([sigmoid_forward(x) for x in xs])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_sigmoid_backward_scalar(self):
        """Test that sigmoid_backward correctly applies the Sigmoid function for a scalar.

        :return: None
        """
        x = 0
        expected = 0.25
        result = sigmoid_backward(x)
        self.assertTrue(math.isclose(result, expected, rel_tol=1e-6))

    def test_sigmoid_backward_array(self):
        """Test that sigmoid_backward correctly applies the Sigmoid function for an array.

        :return: None
        """
        xs = np.array([-1.0, 0.0, 1.0])
        forward_vals = np.array([sigmoid_forward(x) for x in xs])
        expected = forward_vals * (1 - forward_vals)
        result = np.array([sigmoid_backward(x) for x in xs])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
