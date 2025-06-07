import numpy as np

from utilities.accuracy import get_accuracy
from utilities.cross_entropy_loss import cross_entropy_loss as ce_loss
from utilities.softmax import calculate_softmax as softmax
from utilities.relu import relu_forward, relu_backward
import sys

np.random.seed(1024)
np.set_printoptions(threshold=sys.maxsize)


class NeuralNetworks:
    def __init__(self, input_size=28 * 28, hidden_dim=128, output_size=10):
        """Initialize the Neural Network with specified dimensions.

        :param input_size: Size of the input layer, defaults to 28*28 (MNIST image size)
        :param hidden_dim: Size of the hidden layer, defaults to 128
        :param output_size: Size of the output layer, defaults to 10 (number of classes in MNIST)
        """

        self.params = dict()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.init_weights()
        self.init_gradients()
        self.logits = None

    def init_weights(self):
        """Initialize the weights and biases of the neural network.

        Uses He initialization for weights and zeros for biases.
        """

        self.params["w1"] = (
            2
            / np.sqrt(self.input_size)
            * np.random.randn(self.input_size, self.hidden_dim)
        )
        self.params["b1"] = np.zeros(self.hidden_dim)
        self.params["w2"] = (
            2
            / np.sqrt(self.input_size)
            * np.random.randn(self.hidden_dim, self.output_size)
        )
        self.params["b2"] = np.zeros(self.output_size)

    def init_gradients(self):
        """Initialize the gradient matrices and vectors to zeros.

        Creates gradient placeholders for weights and biases that will be updated during training.
        """

        self.params["grad_w1"] = np.zeros_like((self.input_size, self.hidden_dim))
        self.params["grad_b1"] = np.zeros(self.hidden_dim)
        self.params["grad_w2"] = np.zeros_like((self.hidden_dim, self.output_size))
        self.params["grad_b2"] = np.zeros(self.output_size)

    def forward(self, x, y, mode="train"):
        """Perform the forward pass of the neural network.

        :param x: Input data with shape (batch_size, 28x28)
        :param y: Ground truth labels with shape (batch_size)
        :param mode: Operation mode, either 'train' or 'test'
        :return: Tuple of (loss, accuracy)
        """
        batch_size = x.shape[0]
        layer_1_output = x @ self.params["w1"] + self.params["b1"]
        relu_output = relu_forward(layer_1_output)
        layer_2_output = relu_output @ self.params["w2"] + self.params["b2"]
        self.logits = layer_2_output
        softmax_output = softmax(layer_2_output)
        accuracy = get_accuracy(softmax_output, y)

        loss = ce_loss(y, softmax_output)
        if mode != "train":
            return loss, accuracy

        ## partials for ce loss and softmax function.
        ##  This gives the gradient of the loss wrt. the softmax inputs
        ohe = np.zeros((batch_size, self.output_size))
        batch_indices = np.arange(0, batch_size)
        ohe[batch_indices, y] = 1
        loss_sm_gradient = softmax_output - ohe

        self.params["grad_w2"] = (1 / batch_size) * relu_output.T @ loss_sm_gradient
        self.params["grad_b2"] = (1 / batch_size) * np.sum(loss_sm_gradient, axis=0)
        grad_layer_2_x_input = loss_sm_gradient @ self.params["w2"].T
        relu_der = grad_layer_2_x_input * relu_backward(layer_1_output)
        self.params["grad_w1"] = (1 / batch_size) * x.T @ relu_der
        self.params["grad_b1"] = (1 / batch_size) * np.sum(relu_der, axis=0)
        return loss, accuracy
