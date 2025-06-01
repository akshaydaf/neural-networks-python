from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

np.random.seed(42)


def get_data():
    """
    Creates Train and Test CSVs for batching into NN Training
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
    """
    Split data for training and testing
    :param data: ndarray, data to split
    :param batch_size: int, size of each batch
    :return: chunks:
    """
    chunks = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    return chunks


def get_batches(is_get_train=True, should_shuffle=True, batch_size=32):
    """
    Splits image and label data into chunks and returns it
    :param is_get_train: boolean, should be chunking train or test
    :param should_shuffle: boolean, should be shuffling
    :param batch_size: int, size of each chunk
    :return: tuple(list, list), (image_splits, label_splits)
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
