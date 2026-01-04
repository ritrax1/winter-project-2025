import numpy as np
from sklearn.metrics import mean_squared_error
from models import linear_model, random_forest, neural_net
from baseline import evaluate_baseline

X = np.load("data/raw/power.npy")
y = np.load("data/raw/risk.npy")

train_mask = X[:, 0] <= 300
test_mask = X[:, 0] > 300

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

results = {}

b_train_mse, b_test_mse, b_preds = evaluate_baseline(
    X_train, y_train, X_test, y_test
)

results["Physics baseline"] = {
    "train_mse": b_train_mse,
    "test_mse": b_test_mse,
    "test_pred": b_preds
}

models = {
    "Linear regression": linear_model(),
    "Random forest": random_forest(),
    "Neural network": neural_net()
}

for name, model in models.items():
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results[name] = {
        "train_mse": mean_squared_error(y_train, train_pred),
        "test_mse": mean_squared_error(y_test, test_pred),
        "test_pred": test_pred
    }

np.save("data/raw/results.npy", results, allow_pickle=True)

for name, r in results.items():
    print(f"{name}")
    print(f"  Train MSE: {r['train_mse']:.4f}")
    print(f"  Test  MSE: {r['test_mse']:.4f}")