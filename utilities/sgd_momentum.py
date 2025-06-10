from utilities._base_optimizer import _BaseOptimizer


class SGDMomentum(_BaseOptimizer):
    def __init__(self, momentum, **kwargs):
        """Initialize the SGDMomentum Optimizer with specified parameters.

        :param momentum: float, momentum parameter that dictates weight of previous velocity values.
        """
        super().__init__(**kwargs)
        self.momentum = momentum
        self.prev_velocity_w1 = 0.0
        self.prev_velocity_w2 = 0.0

    def update(self, model):
        """Update the model parameters using calculated gradients.

        :param model: instance of NeuralNetworks class
        :return: None
        """
        self.apply_regularization(model)
        self.prev_velocity_w1 = (
            self.momentum * self.prev_velocity_w1 + self.lr * model.params["grad_w1"]
        )
        model.params["w1"] -= self.prev_velocity_w1

        self.prev_velocity_w2 = (
            self.momentum * self.prev_velocity_w2 + self.lr * model.params["grad_w2"]
        )
        model.params["w2"] -= self.prev_velocity_w2
