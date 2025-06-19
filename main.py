from data import load_mnist
from model import build_model
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_model()
    train_model(x_train, y_train, model)
    evaluate_model(x_test, y_test, model)