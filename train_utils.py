import pandas as pd
import mlflow
import mlflow.keras

def train_model(x_train, y_train, model, epochs=5, batch_size=64, learning_rate=0.001):
    mlflow.set_experiment('mnist_classification')
    with mlflow.start_run(run_name=f"bs={batch_size}_lr={learning_rate}"):
        mlflow.set_tag("model", "CNN")
        mlflow.set_tag("dataset", "MNIST")

        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Train the model
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        # Log metrics per epoch
        for epoch in range(epochs):
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("history_logs.csv", index=False)
        mlflow.log_artifact("history_logs.csv")

        # Save model
        model.save("mnist_model.h5")
        mlflow.log_artifact("mnist_model.h5")