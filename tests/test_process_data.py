import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from data.process_data import get_batches


class TestGetBatches(unittest.TestCase):
    @patch("pandas.read_csv")
    def test_get_batches(self, mock_read_csv):
        """Test that get_batches correctly splits data into batches of the specified size.
        
        :param mock_read_csv: Mock object for pandas.read_csv
        :return: None
        """
        # Mock MNIST-style data: 23 samples, 784 features + 1 label
        dummy_data = np.hstack(
            [
                np.arange(23).reshape(-1, 1),  # labels
                np.random.rand(23, 784),  # image data
            ]
        )
        dummy_df = pd.DataFrame(dummy_data)
        mock_read_csv.return_value = dummy_df

        batch_size = 5
        image_batches, label_batches = get_batches(
            is_get_train=True, should_shuffle=False, batch_size=batch_size
        )

        expected_num_batches = int(np.ceil(23 / batch_size))
        self.assertEqual(len(image_batches), expected_num_batches)
        self.assertEqual(len(label_batches), expected_num_batches)

        for i in range(expected_num_batches - 1):  # All but last should be full
            self.assertEqual(image_batches[i].shape, (batch_size, 784))
            self.assertEqual(label_batches[i].shape, (batch_size,))

        self.assertLessEqual(image_batches[-1].shape[0], batch_size)
        self.assertEqual(image_batches[-1].shape[1], 784)
        self.assertEqual(label_batches[-1].shape[0], image_batches[-1].shape[0])


if __name__ == "__main__":
    unittest.main()
