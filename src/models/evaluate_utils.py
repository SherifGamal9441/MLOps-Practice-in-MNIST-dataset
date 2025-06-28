import json

def evaluate_model(x_test, y_test, model):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    with open("logs/metrics.json", "w") as f:
        json.dump({"loss": float(test_loss), "accuracy": float(test_acc)}, f)