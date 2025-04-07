import numpy as np


def calculate_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]