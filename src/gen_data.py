import numpy as np

POWER_MIN = 50      # W
POWER_MAX = 500     # W

ALPHA = 1.4         # debris exponent
BETA = 0.7          # mitigation exponent
GAMMA = 0.8         # gas flow scaling

KD = 1.0            # debris scale
KM = 1.0            # mitigation scale
KF = 0.05           # gas flow scale

EPSILON = 1e-3
NOISE_STD = 0.03

def debris_generation(power):
    return KD * power ** ALPHA


def gas_flow(power):
    return KF * power ** GAMMA


def mitigation(flow):
    return KM * flow ** BETA


def contamination_risk(power):
    D = debris_generation(power)
    F = gas_flow(power)
    M = mitigation(F)

    raw_risk = D / (M + EPSILON)
    risk = 1 - np.exp(-raw_risk)

    return risk

def generate_dataset(n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)

    power = rng.uniform(POWER_MIN, POWER_MAX, size=n_samples)
    risk = contamination_risk(power)

    noise = rng.normal(0, NOISE_STD, size=n_samples)
    risk_noisy = np.clip(risk + noise, 0.0, 1.0)

    return power.reshape(-1, 1), risk_noisy

if __name__ == "__main__":
    X, y = generate_dataset()

    np.save("data/raw/power.npy", X)
    np.save("data/raw/risk.npy", y)

    print("Dataset generated:")
    print(f"Samples: {len(X)}")
    print(f"Power range: {X.min():.1f}–{X.max():.1f} W")