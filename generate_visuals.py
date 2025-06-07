import matplotlib.pyplot as plt
import seaborn as sns
from data.process_data import get_batches
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def generate_heatmaps(data, title):
    """Display a heatmap visualization of the provided data.

    :param data: ndarray, the data to visualize as a heatmap
    :param title: string, the title for the heatmap plot
    :return: None
    """
    ax = sns.heatmap(data)
    plt.title(title)
    plt.show()


def generate_bar_graph(data, title, x_label, y_label):
    """Display a bar graph of the provided data.

    :param data: list or array, the data to visualize as bars
    :param title: string, the title for the bar graph
    :param x_label: string, the label for the x-axis
    :param y_label: string, the label for the y-axis
    :return: None
    """
    plt.bar(list(range(len(data))), data, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def generate_tsne_clusters(trainer):
    """Display t-SNE visualization of the model's final layer outputs.

    Creates a 2D visualization of the high-dimensional representation in the
    model's final layer (logits) using t-SNE dimensionality reduction.

    :param trainer: Trainer instance with a trained model
    :return: None
    """
    x_test, y_test = get_batches(
        is_get_train=False, should_shuffle=True, batch_size=60000
    )
    for index in range(len(x_test)):
        trainer.model.forward(np.array(x_test[index]), np.array(y_test[index]), "test")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(trainer.model.logits)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=y_test, cmap="tab10", alpha=0.6
    )
    plt.colorbar(scatter, ticks=range(10))
    plt.title("t-SNE of Final Layer Logits (Before Softmax)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()


def generate_confusion_matrix(trainer):
    """Display a confusion matrix for the model's predictions.

    Evaluates the model on the test dataset and creates a confusion matrix
    to visualize the model's classification performance across all classes.

    :param trainer: Trainer instance with a trained model
    :return: None
    """
    x_test, y_test = get_batches(
        is_get_train=False, should_shuffle=True, batch_size=10000
    )
    for index in range(len(x_test)):
        trainer.model.forward(np.array(x_test[index]), np.array(y_test[index]), "test")

    y_pred = np.argmax(trainer.model.logits, axis=1)

    confusion_matrix_result = confusion_matrix(np.squeeze(np.array(y_test)), y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result)
    disp.plot()
    plt.show()
