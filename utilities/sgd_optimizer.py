from utilities._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def update(self, model):
        """Update the model parameters using calculated gradients.

        :param model: instance of NeuralNetworks class
        :return: None
        """
        self.apply_regularization(model)
        model.params["w1"] -= self.lr * model.params["grad_w1"]
        model.params["w2"] -= self.lr * model.params["grad_w2"]
