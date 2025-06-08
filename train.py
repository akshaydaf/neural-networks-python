from data.process_data import get_batches
from model.neural_networks import NeuralNetworks
from utilities.optimizer import Optimizer
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import yaml
import copy


def load_config(file_path):
    """Load configuration from a YAML file.

    :param file_path: string, path to the YAML configuration file
    :return: dict containing configuration parameters, or None if loading fails
    """
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None


class Trainer:
    def __init__(self, config_file_name="configs/config.yaml", activation="relu"):
        """Initialize the Trainer.

        :param config_file_name: Configuration file string path
        :param activation: Type of activation, defaults to relu
        """
        self.training_loss_across_epochs = []
        self.validation_loss_across_epochs = []
        self.training_accuracy_across_epochs = []
        self.validation_accuracy_across_epochs = []
        self.gradient_absolute_values = []
        self.model = NeuralNetworks(activation=activation)
        self.initial_weight_values_w1 = copy.deepcopy(self.model.params["w1"])
        self.raw_image_map = np.zeros(self.model.input_size)
        self.config_file_name = config_file_name

    def train(self):
        """Execute the training loop for the neural network.

        Loads configuration, prepares data batches, initializes the optimizer,
        and runs the training loop for the specified number of epochs.
        Updates internal tracking variables for loss and accuracy metrics.

        :return: None
        """
        config = load_config(self.config_file_name)

        x, y = get_batches(
            is_get_train=True, should_shuffle=True, batch_size=config["batch_size"]
        )
        x_test, y_test = get_batches(
            is_get_train=False, should_shuffle=True, batch_size=config["batch_size"]
        )
        optimizer = Optimizer(
            learning_rate=config["learning_rate"],
            regularization_coeff=config["reg"],
            mode=config["mode"],
        )

        for epoch in tqdm(
            range(config["epochs"]),
            desc="Training and Validation by Epoch",
            bar_format="🟩{desc}: {bar}🟥 {percentage:3.0f}%",
        ):
            epoch_training_loss = 0.0
            epoch_training_accuracy = 0.0

            epoch_validation_loss = 0.0
            epoch_validation_accuracy = 0.0
            total_samples = 0
            total_gradient_sum_per_epoch = 0
            for index in range(len(x)):
                optimizer.zero_grad(self.model)
                loss, acc = self.model.forward(np.array(x[index]), np.array(y[index]))
                epoch_training_accuracy += acc * len(x[index])
                epoch_training_loss += loss

                # Used for later analysis, not directly relevant to training
                self.raw_image_map += np.sum(x[index], axis=0)
                total_gradient_sum_per_epoch += np.sum(
                    np.absolute(self.model.params["grad_w1"])
                ) + np.sum(np.absolute(self.model.params["grad_w2"]))
                total_samples += len(x[index])
                # Done with analysis block

                optimizer.update(self.model)

            epoch_training_loss /= len(x)
            epoch_training_accuracy /= total_samples
            self.training_loss_across_epochs.append(epoch_training_loss)
            self.training_accuracy_across_epochs.append(epoch_training_accuracy)

            # Used for later analysis, not directly relevant to training
            total_gradient_sum_per_epoch /= len(x)
            self.raw_image_map /= len(x)
            self.gradient_absolute_values.append(total_gradient_sum_per_epoch)
            # Done with analysis block

            total_samples = 0
            for index in range(len(x_test)):
                loss, acc = self.model.forward(
                    np.array(x_test[index]), np.array(y_test[index]), "test"
                )

                epoch_validation_accuracy += acc * len(x_test[index])
                total_samples += len(x_test[index])
                epoch_validation_loss += loss
            epoch_validation_loss /= len(x_test)
            epoch_validation_accuracy /= total_samples
            self.validation_loss_across_epochs.append(epoch_validation_loss)
            self.validation_accuracy_across_epochs.append(epoch_validation_accuracy)

    def generate_plots(self):
        """Generate and display plots for training and validation metrics.

        Creates plots showing the loss and accuracy curves for both
        training and validation data across all training epochs.

        :return: None
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
