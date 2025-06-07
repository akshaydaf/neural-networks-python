import numpy as np


def get_accuracy(x, y):
    """Calculate the accuracy of model predictions.

    :param x: ndarray (B, num_classes), the softmax probabilities computed by the network
    :param y: ndarray (B,), actual values
    :return: an accuracy percentage
    """
    max_index = np.argmax(x, axis=1)
    num_same = np.sum(max_index == y)
    return num_same / len(y)
