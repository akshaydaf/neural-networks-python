import unittest
import numpy as np
from utilities.accuracy import get_accuracy


class TestGetAccuracy(unittest.TestCase):

    def test_all_correct(self):
        x = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        y = np.array([1, 0, 1])
        self.assertAlmostEqual(get_accuracy(x, y), 1.0)

    def test_all_incorrect(self):
        x = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        y = np.array([1, 0, 1])
        self.assertAlmostEqual(get_accuracy(x, y), 0.0)

    def test_half_correct(self):
        x = np.array([[0.9, 0.1], [0.2, 0.8]])
        y = np.array([0, 0])
        self.assertAlmostEqual(get_accuracy(x, y), 0.5)


if __name__ == "__main__":
    unittest.main()
