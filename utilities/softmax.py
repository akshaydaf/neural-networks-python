import numpy as np


def calculate_softmax(x):
    """
    Computes Softmax probabilities given some input x
    :param x: variable length input
    :return: variable length output, same size as input
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
