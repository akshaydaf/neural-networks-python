import numpy as np


def cross_entropy_loss(y, x_pred):
    """Compute cross-entropy loss between true labels and predicted probabilities.

    :param y: ndarray (batch_size,), ground truth labels
    :param x_pred: ndarray (batch_size, class_size), predicted probabilities
    :return: float, the average cross-entropy loss
    """
    eps = 1e-12
    predictions = np.clip(x_pred, eps, 1.0 - eps)
    return (-1 / len(y)) * np.sum(np.log(predictions[np.arange(len(y)), y]))
