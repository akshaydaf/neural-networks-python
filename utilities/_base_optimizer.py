import numpy as np


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, regularization_coeff=1e-4, mode="l2"):
        """Initialize the Base Optimizer with specified parameters.

        :param learning_rate: float, learning rate for optimizer
        :param regularization_coeff: float, regularization coefficient for weight decay
        :param mode: string, type of regularization to apply, defaults to "l2"
        """
        self.lr = learning_rate
        self.reg_penalty = regularization_coeff
        self.reg_mode = mode

    def update(self, model):
        """Update the model parameters using calculated gradients.

        :param model: instance of NeuralNetworks class
        :return: None
        """
        raise NotImplementedError("Each Optimizer must implement its own update method")

    def apply_regularization(self, model):
        """Apply regularization to the model's gradients.

        :param model: instance of NeuralNetworks class
        :return: None
        """
        if self.reg_mode == "l2":
            model.params["grad_w1"] += self.reg_penalty * model.params["w1"]
            model.params["grad_w2"] += self.reg_penalty * model.params["w2"]
        elif self.reg_mode == "l1":
            model.params["grad_w1"] += self.reg_penalty * np.sign(model.params["w1"])
            model.params["grad_w2"] += self.reg_penalty * np.sign(model.params["w2"])

    @staticmethod
    def zero_grad(model):
        """Reset all gradients in the model to zero.

        :param model: instance of NeuralNetworks class
        :return: None
        """
        model.init_gradients()
