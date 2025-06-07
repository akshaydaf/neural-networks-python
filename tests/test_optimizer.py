import unittest
from unittest.mock import MagicMock
from utilities.optimizer import Optimizer
import numpy as np


class DummyModel:
    def __init__(self):
        self.params = {
            "w1": np.array([[1.0, -2.0], [0.0, 3.0]]),
            "w2": np.array([[0.5, -0.5], [2.0, -1.0]]),
            "grad_w1": np.zeros((2, 2)),
            "grad_w2": np.zeros((2, 2)),
        }

    def init_gradients(self):
        self.params["grad_w1"].fill(0)
        self.params["grad_w2"].fill(0)


class TestOptimizer(unittest.TestCase):
    def test_update_parameters(self):
        """Test that the optimizer correctly updates model parameters using gradients.

        :return: None
        """

        class DummyModel:
            def __init__(self):
                self.params = {"w1": 2.0, "w2": -3.0, "grad_w1": 0.5, "grad_w2": -1.0}

        model = DummyModel()
        optimizer = Optimizer(learning_rate=0.1, regularization_coeff=0)
        optimizer.update(model)

        self.assertAlmostEqual(model.params["w1"], 2.0 - 0.1 * 0.5)
        self.assertAlmostEqual(model.params["w2"], -3.0 - 0.1 * -1.0)

    def test_zero_grad_calls_init_gradients(self):
        """Test that zero_grad method calls the model's init_gradients method.

        :return: None
        """
        # Create a mock model with a MagicMock method
        model = MagicMock()
        Optimizer.zero_grad(model)

        model.init_gradients.assert_called_once()

    def setUp(self):
        """Set up test fixtures for optimizer tests.

        :return: None
        """
        self.model = DummyModel()
        self.reg_strength = 0.1
        self.optimizer = Optimizer(
            learning_rate=1e-3, regularization_coeff=self.reg_strength, mode="l2"
        )

    def test_l2_regularization(self):
        """Test that L2 regularization correctly updates gradients.

        :return: None
        """
        self.optimizer.apply_regularization(self.model)
        expected_grad_w1 = self.reg_strength * self.model.params["w1"]
        expected_grad_w2 = self.reg_strength * self.model.params["w2"]
        np.testing.assert_array_almost_equal(
            self.model.params["grad_w1"], expected_grad_w1
        )
        np.testing.assert_array_almost_equal(
            self.model.params["grad_w2"], expected_grad_w2
        )

    def test_l1_regularization(self):
        """Test that L1 regularization correctly updates gradients.

        :return: None
        """
        self.model.init_gradients()  # Reset gradients
        self.optimizer.reg_mode = "l1"
        self.optimizer.apply_regularization(self.model)
        expected_grad_w1 = self.reg_strength * np.sign(self.model.params["w1"])
        expected_grad_w2 = self.reg_strength * np.sign(self.model.params["w2"])
        np.testing.assert_array_almost_equal(
            self.model.params["grad_w1"], expected_grad_w1
        )
        np.testing.assert_array_almost_equal(
            self.model.params["grad_w2"], expected_grad_w2
        )


if __name__ == "__main__":
    unittest.main()
