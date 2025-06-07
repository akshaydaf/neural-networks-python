import unittest
import numpy as np
from model.neural_networks import NeuralNetworks


class TestNeuralNetwork(unittest.TestCase):
    def test_forward_pass_loss_output(self):
        """Test that the forward pass produces valid loss and accuracy values.
        
        :return: None
        """
        np.random.seed(0)
        batch_size = 4
        input_size = 28 * 28
        num_classes = 10

        x = np.random.rand(batch_size, input_size)
        y = np.random.randint(0, num_classes, size=(batch_size,))

        model = NeuralNetworks(
            input_size=input_size, hidden_dim=128, output_size=num_classes
        )

        loss, acc = model.forward(x, y, mode="train")

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 10.0)
