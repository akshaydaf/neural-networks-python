class Optimizer:
    def __init__(self, learning_rate):
        """
        :param learning_rate: float
        """
        self.lr = learning_rate

    def update(self, model):
        """
        :param model: instance of NeuralNetworks class
        :return: None
        """
        model.params["w1"] -= self.lr * model.params["grad_w1"]
        model.params["w2"] -= self.lr * model.params["grad_w2"]

    @staticmethod
    def zero_grad(model):
        """
        :param model: instance of NeuralNetworks class
        :return: None
        """
        model.init_gradients()
