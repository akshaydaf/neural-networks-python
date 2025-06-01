from sklearn.datasets import fetch_openml
import pandas as pd


def get_data():
    """
    Creates Train and Test CSVs for batching into NN Training
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(int)
    df = pd.DataFrame(data=X)
    df.insert(0, "label", y)
    df_train = df[:60000]
    df_test = df[60000:]

    # Save to CSV
    df_train.to_csv("mnist_train.csv", index=False)
    df_test.to_csv("mnist_test.csv", index=False)

    print("Saved mnist_train.csv and mnist_test.csv!")


def get_batches(is_get_train):
    if is_get_train:
        df = pd.read_csv("mnist_train.csv")
    else:
        df = pd.read_csv("mnist_test.csv")
