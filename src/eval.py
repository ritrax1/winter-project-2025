import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/raw/power.npy")
y = np.load("data/raw/risk.npy")
results = np.load("data/raw/results.npy", allow_pickle=True).item()

train_mask = X[:, 0] <= 300
test_mask = X[:, 0] > 300

X_test = X[test_mask]
y_test = y[test_mask]

# Sort for clean curves
idx = np.argsort(X_test[:, 0])
X_test = X_test[idx]
y_test = y_test[idx]

plt.figure(figsize=(9, 5))

plt.scatter(X, y, s=10, alpha=0.25, label="Synthetic data")

for name, r in results.items():
    plt.plot(
        X_test[:, 0],
        r["test_pred"][idx],
        linewidth=2,
        label=name
    )

plt.axvline(300, color="k", linestyle="--", label="Train/Test boundary")
plt.xlabel("EUV Power (W)")
plt.ylabel("Contamination Risk")
plt.title("Model behavior under extrapolation")
plt.legend()
plt.tight_layout()
plt.savefig("figures/extrapolation_comparison.png")
plt.show()