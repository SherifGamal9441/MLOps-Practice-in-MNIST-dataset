def evaluate_model(x_test,y_test,model):
    "evaluate the model performance"
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")