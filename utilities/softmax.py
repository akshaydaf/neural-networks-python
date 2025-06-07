import numpy as np


def calculate_softmax(x):
    """Compute softmax probabilities from input logits.

    :param x: ndarray of shape (batch_size, num_classes), input logits
    :return: ndarray of same shape as input, containing softmax probabilities
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
