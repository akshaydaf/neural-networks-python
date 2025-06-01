from data.process_data import get_batches
from model.neural_networks import NeuralNetworks
from utilities.optimizer import Optimizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self):
        self.training_loss_across_epochs = []
        self.validation_loss_across_epochs = []
        self.training_accuracy_across_epochs = []
        self.validation_accuracy_across_epochs = []

    def train(self):
        """
        Training loop for neural network
        """
        x, y = get_batches(
            is_get_train=True, should_shuffle=True, batch_size=64
        )  ## TODO: Replace with config
        x_test, y_test = get_batches(
            is_get_train=False, should_shuffle=True, batch_size=64
        )  ## TODO: Replace with config
        model = NeuralNetworks()
        epochs = 10
        optimizer = Optimizer(learning_rate=1e-4)

        for epoch in range(epochs):
            epoch_training_loss = 0.0
            epoch_training_accuracy = 0.0

            epoch_validation_loss = 0.0
            epoch_validation_accuracy = 0.0
            total_samples = 0
            for index in tqdm(range(len(x))):
                optimizer.zero_grad(model)
                loss, acc = model.forward(np.array(x[index]), np.array(y[index]))
                epoch_training_accuracy += acc * len(x[index])
                total_samples += len(x[index])
                epoch_training_loss += loss
                optimizer.update(model)
            epoch_training_loss /= len(x)
            epoch_training_accuracy /= total_samples
            self.training_loss_across_epochs.append(epoch_training_loss)
            self.training_accuracy_across_epochs.append(epoch_training_accuracy)
            total_samples = 0
            for index in tqdm(range(len(x_test))):
                loss, acc = model.forward(
                    np.array(x_test[index]), np.array(y_test[index])
                )

                epoch_validation_accuracy += acc * len(x[index])
                total_samples += len(x[index])
                epoch_validation_loss += loss
            epoch_validation_loss /= len(x_test)
            epoch_validation_accuracy /= total_samples
            self.validation_loss_across_epochs.append(epoch_validation_loss)
            self.validation_accuracy_across_epochs.append(epoch_validation_accuracy)

    def generate_plots(self):
        """
        Generate Plots for the loss and accuracy curves
        """
        train_loss_x_axis = list(range(len(self.training_loss_across_epochs)))
        train_loss_y_axis = self.training_loss_across_epochs

        test_loss_y_axis = self.validation_loss_across_epochs

        train_acc_x_axis = list(range(len(self.training_accuracy_across_epochs)))
        train_acc_y_axis = self.training_accuracy_across_epochs

        test_acc_y_axis = self.validation_accuracy_across_epochs

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

        # Loss subplot
        axs[0].plot(
            train_loss_x_axis,
            train_loss_y_axis,
            label="Training Loss Curve",
            marker="o",
            linestyle="-",
        )
        axs[0].plot(
            train_loss_x_axis,
            test_loss_y_axis,
            label="Validation Loss Curve",
            marker="o",
            linestyle="-",
        )
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training vs. Validation Loss Curves")
        # Accuracy subplot
        axs[1].plot(
            train_acc_x_axis,
            train_acc_y_axis,
            label="Training Accuracy Curve",
            marker="o",
            linestyle="-",
        )
        axs[1].plot(
            train_acc_x_axis,
            test_acc_y_axis,
            label="Validation Accuracy Curve",
            marker="o",
            linestyle="-",
        )
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].set_title("Training vs. Validation Accuracy")

        plt.tight_layout()
        plt.legend(["Training Curve", "Validation Curve"])

        plt.show()
