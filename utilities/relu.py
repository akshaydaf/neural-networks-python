
def relu_forward(x):
    """
    Forward Pass of ReLU
    :param x: variable length input
    :return: variable length output in the same shape as input
    """
    mask = x <= 0
    x[mask] = 0
    return x


def relu_backward(x):
    """
    Derivative of the ReLU function
    :param x: variable length input
    :return: variable length output in the same shape as input
    """
    mask = x <= 0
    x[mask] = 0
    mask = x > 0
    x[mask] = 1
    return x
