import math


def sigmoid_forward(x):
    """Apply the Sigmoid activation function to the input.

    :param x: ndarray of any shape, input data
    :return: ndarray of the same shape as input, with Sigmoid applied
    """
    return 1 / (1 + math.e ** (-x))


def sigmoid_backward(x):
    """Calculate the derivative of the Sigmoid function.

    :param x: ndarray of any shape, input data
    :return: ndarray of the same shape as input, with derivative of Sigmoid applied
    """

    return sigmoid_forward(x) * (1 - sigmoid_forward(x))
