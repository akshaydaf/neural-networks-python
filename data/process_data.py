from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

np.random.seed(42)


def get_data():
    """Download and create train and test CSV files for neural network training.
    
    Downloads the MNIST dataset and splits it into training and testing sets,
    then saves them as CSV files for later use.
    
    :return: None
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(int)
    df = pd.DataFrame(data=X)
    df.insert(0, "label", y)
    df_train = df[:60000]
    df_test = df[60000:]

    # Save to CSV
    df_train.to_csv("mnist_train.csv", index=False)
    df_test.to_csv("mnist_test.csv", index=False)

    print("Saved mnist_train.csv and mnist_test.csv!")


def split_data(data, batch_size):
    """Split data into batches of specified size.
    
    :param data: ndarray, data to split
    :param batch_size: int, size of each batch
    :return: list of data chunks, each of size batch_size (except possibly the last one)
    """
    chunks = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    return chunks


def get_batches(is_get_train=True, should_shuffle=True, batch_size=32):
    """Load and split image and label data into batches.
    
    :param is_get_train: boolean, whether to use training data (True) or test data (False)
    :param should_shuffle: boolean, whether to shuffle the data before batching
    :param batch_size: int, size of each batch
    :return: tuple(list, list), containing (image_batches, label_batches)
    """
    if is_get_train:
        df = pd.read_csv("data/mnist_train.csv")
    else:
        df = pd.read_csv("data/mnist_test.csv")
    data = df.to_numpy()
    if should_shuffle:
        np.random.shuffle(data)
    image_data = data[:, 1:]
    label_data = data[:, 0]
    image_splits = split_data(image_data, batch_size)
    label_splits = split_data(label_data, batch_size)
    return image_splits, label_splits
