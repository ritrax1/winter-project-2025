from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy

def linear_model():
    return LinearRegression()


def random_forest():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

class BoundedMLP(MLPRegressor):
    def predict(self, X):
        raw = super().predict(X)
        return numpy.clip(raw, 0.0, 1.0)


def neural_net():
    return BoundedMLP(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        alpha=1e-4,
        max_iter=1500,
        early_stopping=True,
        random_state=42
    )