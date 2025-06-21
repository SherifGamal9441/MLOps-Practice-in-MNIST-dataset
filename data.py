import numpy as np

def load_mnist():
    "Loads training data and preprocess it"
    data = np.load("mnist_data.npz")
    x_train, y_train , x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)