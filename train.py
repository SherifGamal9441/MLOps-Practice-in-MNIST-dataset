import pandas as pd
def train_model(x_train, y_train, model):
    "Training the model and saving it to .csv file"
    history = model.fit(x_train, y_train, epochs=5, batch_size=64,validation_split=0.1)
    history_csv = pd.DataFrame(history.history)
    history_csv.to_csv('history_logs.csv', index = False)