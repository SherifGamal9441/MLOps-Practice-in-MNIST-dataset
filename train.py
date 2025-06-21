import pandas as pd
import mlflow
import mlflow.keras

def train_model(x_train, y_train, model, epochs=5, batch_size=64, learning_rate=0.001):
    "Training the model and saving it to .csv file"

    mlflow.set_experiment('mnist_classification')
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Train the model
        history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_split=0.1)

        # log all validation metrics
        for epoch in range(epochs):
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Save training history
        history_csv = pd.DataFrame(history.history)
        history_csv.to_csv('history_logs.csv', index = False)
        mlflow.log_artifact("history_logs.csv")

        # Log metrics from the last epoch
        final_metrics = history.history
        mlflow.log_metric("val_accuracy", final_metrics["val_accuracy"][-1])
        mlflow.log_metric("val_loss", final_metrics["val_loss"][-1])

        # Log the full model
        mlflow.keras.log_model(model, "mnist_model")

        # Save the model
        model.save("mnist_model.h5")