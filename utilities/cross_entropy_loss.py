import numpy as np


def cross_entropy_loss(y, x_pred):
    return (-1/len(y)) * np.sum(np.log(x_pred[np.arange(len(y)), y]))
