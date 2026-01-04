import numpy as np
from sklearn.metrics import mean_squared_error

def physics_baseline(power, a=1.4, scale=1.0):
    raw = scale * power ** a
    return 1 - np.exp(-raw)

def evaluate_baseline(X_train, y_train, X_test, y_test):
    preds_train = physics_baseline(X_train[:, 0])
    preds_test = physics_baseline(X_test[:, 0])

    train_mse = mean_squared_error(y_train, preds_train)
    test_mse = mean_squared_error(y_test, preds_test)

    return train_mse, test_mse, preds_test