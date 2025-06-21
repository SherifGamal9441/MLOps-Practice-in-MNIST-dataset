from data import load_mnist
from model import build_model
from train import train_model
from evaluate import evaluate_model
import yaml
if __name__ == "__main__":
    
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    batch_size = params["batch_size"]
    epochs = params["epochs"]
    learning_rate = params["learning_rate"]

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_model(learning_rate=learning_rate)
    train_model(x_train, y_train, model, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
    evaluate_model(x_test, y_test, model)