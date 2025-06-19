from data import load_mnist

def test_load_mnist():
    "Testing the model for errors in shape"
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    print('Tests successful')