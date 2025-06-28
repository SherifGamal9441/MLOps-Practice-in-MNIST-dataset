from src.data.data import load_mnist
from tensorflow.keras.models import load_model
from src.models.evaluate_utils import evaluate_model

if __name__ == "__main__":
    _, (x_test, y_test) = load_mnist()
    model = load_model("mnist_model.h5")
    evaluate_model(x_test, y_test, model)
