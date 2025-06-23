from data import load_mnist
from model import build_model
from train_utils import train_model
import yaml

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    (x_train, y_train), _ = load_mnist()
    model = build_model()
    train_model(
        x_train, y_train, model,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"]
    )