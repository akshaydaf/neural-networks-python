import unittest
import numpy as np
from model.neural_networks import NeuralNetworks


class TestNeuralNetwork(unittest.TestCase):
    def test_forward_pass_loss_output_all_activations(self):
        """Test that the forward pass produces valid loss and accuracy values.

        :return: None
        """
        for activation in ["relu", "sigmoid"]:
            with self.subTest(activation=activation):
                model = NeuralNetworks(
                    input_size=784,
                    hidden_dim=128,
                    output_size=10,
                    activation=activation,
                )
                x = np.random.rand(4, 784)
                y = np.random.randint(0, 10, size=(4,))
                loss, acc = model.forward(x, y, mode="train")
                self.assertIsInstance(loss, float)
                self.assertGreater(loss, 0.0)
                self.assertLess(loss, 10.0)
                self.assertIsInstance(acc, float)
                self.assertGreaterEqual(acc, 0.0)
                self.assertLessEqual(acc, 1.0)
