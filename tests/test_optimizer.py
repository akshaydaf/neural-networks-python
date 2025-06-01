import unittest
from unittest.mock import MagicMock
from utilities.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):
    def test_update_parameters(self):
        class DummyModel:
            def __init__(self):
                self.params = {"w1": 2.0, "w2": -3.0, "grad_w1": 0.5, "grad_w2": -1.0}

        model = DummyModel()
        optimizer = Optimizer(learning_rate=0.1)
        optimizer.update(model)

        self.assertAlmostEqual(model.params["w1"], 2.0 - 0.1 * 0.5)
        self.assertAlmostEqual(model.params["w2"], -3.0 - 0.1 * -1.0)

    def test_zero_grad_calls_init_gradients(self):
        # Create a mock model with a MagicMock method
        model = MagicMock()
        Optimizer.zero_grad(model)

        model.init_gradients.assert_called_once()


if __name__ == "__main__":
    unittest.main()
