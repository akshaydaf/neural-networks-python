import numpy as np
import unittest
from utilities.cross_entropy_loss import cross_entropy_loss


class TestNumpyCrossEntropyLoss(unittest.TestCase):
    def test_cross_entropy_loss(self):
        x_pred = np.array([
            [0.75, 0.15, 0.10],
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85],
            [0.33, 0.34, 0.33]
        ], dtype=np.float32)
        y = np.array([0, 1, 2, 1], dtype=np.int32)
        expected_loss = np.mean([
            -np.log(0.75),
            -np.log(0.80),
            -np.log(0.85),
            -np.log(0.34)
        ])
        loss = cross_entropy_loss(y, x_pred)

        self.assertAlmostEqual(loss, expected_loss, places=5)
