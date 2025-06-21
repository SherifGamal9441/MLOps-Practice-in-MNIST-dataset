from data import load_mnist
from model import build_model
from train import train_model
from evaluate import evaluate_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_model(learning_rate=args.learning_rate)
    train_model(x_train, y_train, model, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate)
    evaluate_model(x_test, y_test, model)