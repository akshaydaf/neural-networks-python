
def relu_forward(x):
    mask = x <= 0
    x[mask] = 0
    return x


def relu_backward(x):
    mask = x <= 0
    x[mask] = 0
    mask = x > 0
    x[mask] = 1
    return x
