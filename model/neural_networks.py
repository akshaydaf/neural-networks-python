import numpy as np

from utilities.accuracy import get_accuracy
from utilities.cross_entropy_loss import cross_entropy_loss as ce_loss
from utilities.sigmoid import sigmoid_forward, sigmoid_backward
from utilities.softmax import calculate_softmax as softmax
from utilities.relu import relu_forward, relu_backward
import sys

np.random.seed(1024)
np.set_printoptions(threshold=sys.maxsize)


class NeuralNetworks:
    def __init__(
        self, input_size=28 * 28, hidden_dim=128, output_size=10, activation="relu"
    ):
        """Initialize the Neural Network with specified dimensions.

        :param input_size: Size of the input layer, defaults to 28*28 (MNIST image size)
        :param hidden_dim: Size of the hidden layer, defaults to 128
        :param output_size: Size of the output layer, defaults to 10 (number of classes in MNIST)
        :param activation: Type of activation, defaults to relu
        """

        self.params = dict()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.logits = None
        self.activation = activation
        self.activation_function_forward = (
            relu_forward if activation == "relu" else sigmoid_forward
        )
        self.activation_function_backward = (
            relu_backward if activation == "relu" else sigmoid_backward
        )

        self.init_weights()
        self.init_gradients()

    def init_weights(self):
        """Initialize the weights and biases of the neural network.

        Uses He initialization for weights and zeros for biases if using ReLU activation.
        Uses Normalized Xavier initialization for weights and zeros for biases if using Sigmoid activation.
        """
        if self.activation == "relu":
            self.params["w1"] = (
                2
                / np.sqrt(self.input_size)
                * np.random.randn(self.input_size, self.hidden_dim)
            )

            self.params["w2"] = (
                2
                / np.sqrt(self.input_size)
                * np.random.randn(self.hidden_dim, self.output_size)
            )

        else:
            self.params["w1"] = np.random.uniform(
                (-(6**0.5)) / ((self.input_size + self.hidden_dim) ** 0.5),
                (6**0.5) / ((self.input_size + self.hidden_dim) ** 0.5),
                (self.input_size, self.hidden_dim),
            )
            self.params["w2"] = np.random.uniform(
                (-(6**0.5)) / ((self.hidden_dim + self.output_size) ** 0.5),
                (6**0.5) / ((self.hidden_dim + self.output_size) ** 0.5),
                (self.hidden_dim, self.output_size),
            )

        self.params["b1"] = np.zeros(self.hidden_dim)
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
        activation_output = self.activation_function_forward(layer_1_output)
        self.logits = activation_output @ self.params["w2"] + self.params["b2"]
        softmax_output = softmax(self.logits)
        accuracy = get_accuracy(softmax_output, y)

        loss = ce_loss(y, softmax_output)
        if mode != "train":
            return loss, accuracy

        # Partials for ce loss and softmax function. This gives the gradient of the loss with respect to the softmax
        # inputs
        ohe = np.zeros((batch_size, self.output_size))
        batch_indices = np.arange(0, batch_size)
        ohe[batch_indices, y] = 1
        loss_sm_gradient = softmax_output - ohe

        self.params["grad_w2"] = (
            (1 / batch_size) * activation_output.T @ loss_sm_gradient
        )
        self.params["grad_b2"] = (1 / batch_size) * np.sum(loss_sm_gradient, axis=0)
        grad_layer_2_x_input = loss_sm_gradient @ self.params["w2"].T
        activation_der = grad_layer_2_x_input * self.activation_function_backward(
            layer_1_output
        )
        self.params["grad_w1"] = (1 / batch_size) * x.T @ activation_der
        self.params["grad_b1"] = (1 / batch_size) * np.sum(activation_der, axis=0)
        return loss, accuracy
