import numpy as np


def cross_entropy_loss(y, x_pred):
    """
    Computes CE Loss given y and x_predicted
    :param y: (batch_size,)
    :param x_pred: (batch_size, class_size)
    :return: float
    """
    return (-1/len(y)) * np.sum(np.log(x_pred[np.arange(len(y)), y]))
