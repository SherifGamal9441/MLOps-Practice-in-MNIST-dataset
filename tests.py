def test_data_shapes():
    import numpy as np
    x_train = np.random.rand(60000, 28, 28)
    y_train = np.random.randint(0, 10, size=(60000,))
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)