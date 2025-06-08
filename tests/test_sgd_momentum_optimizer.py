import unittest
import numpy as np
from utilities.sgd_momentum import SGDMomentum


class DummyModel:
    """A dummy model class for testing the SGDMomentum optimizer.
    
    This class simulates a neural network model with parameters and gradients
    for testing the SGDMomentum optimizer's functionality.
    """
    def __init__(self):
        self.params = {
            "w1": np.array([1.0, 2.0]),
            "w2": np.array([3.0, 4.0]),
            "grad_w1": np.array([0.1, 0.1]),
            "grad_w2": np.array([0.2, 0.2]),
        }

    def init_gradients(self):
        self.params["grad_w1"] = np.zeros_like(self.params["w1"])
        self.params["grad_w2"] = np.zeros_like(self.params["w2"])


class TestSGDMomentum(unittest.TestCase):
    """Test suite for the SGDMomentum optimizer implementation."""
    
    def test_single_update(self):
        """Test a single parameter update with the SGDMomentum optimizer.
        
        This test verifies that the SGDMomentum optimizer correctly updates
        model parameters after a single optimization step, checking that:
        1. The velocity is correctly calculated as learning_rate * gradient
        2. The parameters are updated by subtracting the velocity
        """
        model = DummyModel()
        optimizer = SGDMomentum(momentum=0.9)
        optimizer.lr = 0.01

        optimizer.update(model)

        expected_velocity_w1 = 0.01 * model.params["grad_w1"]
        expected_velocity_w2 = 0.01 * model.params["grad_w2"]

        np.testing.assert_almost_equal(
            model.params["w1"], np.array([1.0, 2.0]) - expected_velocity_w1
        )
        np.testing.assert_almost_equal(
            model.params["w2"], np.array([3.0, 4.0]) - expected_velocity_w2
        )

    def test_momentum_accumulates(self):
        """Test that momentum properly accumulates over multiple updates.
        
        This test verifies that the SGDMomentum optimizer correctly:
        1. Accumulates velocity across multiple update steps
        2. Applies the momentum factor to the previous velocity
        3. Adds the gradient contribution to the velocity
        4. Updates the model parameters using the accumulated velocity
        """
        model = DummyModel()
        optimizer = SGDMomentum(momentum=0.9)
        optimizer.lr = 0.01

        # First update
        optimizer.update(model)
        # Save weights and velocities
        first_velocity_w1 = optimizer.prev_velocity_w1.copy()
        first_velocity_w2 = optimizer.prev_velocity_w2.copy()
        first_w1 = model.params["w1"].copy()
        first_w2 = model.params["w2"].copy()

        # Second update with same gradients
        optimizer.update(model)

        # Expected velocities: v = 0.9 * v + 0.01 * grad
        expected_velocity_w1 = 0.9 * first_velocity_w1 + 0.01 * model.params["grad_w1"]
        expected_velocity_w2 = 0.9 * first_velocity_w2 + 0.01 * model.params["grad_w2"]

        np.testing.assert_almost_equal(optimizer.prev_velocity_w1, expected_velocity_w1)
        np.testing.assert_almost_equal(
            model.params["w1"], first_w1 - expected_velocity_w1
        )

        np.testing.assert_almost_equal(optimizer.prev_velocity_w2, expected_velocity_w2)
        np.testing.assert_almost_equal(
            model.params["w2"], first_w2 - expected_velocity_w2
        )


if __name__ == "__main__":
    unittest.main()
