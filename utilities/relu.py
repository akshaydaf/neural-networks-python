def relu_forward(x):
    """Apply the ReLU activation function to the input.

    :param x: ndarray of any shape, input data
    :return: ndarray of the same shape as input, with ReLU applied
    """
    mask = x <= 0
    x[mask] = 0
    return x


def relu_backward(x):
    """Calculate the derivative of the ReLU function.

    :param x: ndarray of any shape, input data
    :return: ndarray of the same shape as input, with derivative of ReLU applied
    """
    mask = x <= 0
    x[mask] = 0
    mask = x > 0
    x[mask] = 1
    return x
