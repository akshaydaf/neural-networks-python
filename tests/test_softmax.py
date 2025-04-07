import numpy as np
import unittest
from utilities.softmax import calculate_softmax


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.23629739, 0.76370261],
                      [0.67372745, 0.32627255],
                      [0.35677439, 0.64322561],
                      [0.07239128, 0.92760872]])

        out = calculate_softmax(x)

        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)
