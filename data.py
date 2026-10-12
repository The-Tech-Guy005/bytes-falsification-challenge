"""
data.py - Synthetic Dataset Generator with Matched Information Structure

Constructs synthetic data with matched dimensionality (2D) and information structure:
1. Feature A (Simple, 2D): [1D linear signal, 1D independent noise].
2. Feature B (Complex, 2D): [2D raw concentric rings] (non-linear decision boundary).
3. Feature B_control (Diagnostic Control, 2D): [1D normalized radial signal, 1D independent noise].
   Normalization mean and std are calculated strictly on training samples to prevent data leakage.
"""

import numpy as np


def generate_dataset(n_samples=1000, seed=42, test_split=0.3):
    """
    Generates synthetic binary classification data under controlled conditions.

    Parameters:
    -----------
    n_samples : int, total samples (default 1000, 50% class 0, 50% class 1)
    seed : int, deterministic random seed
    test_split : float, fraction of test data (default 0.3)

    Returns:
    --------
    dict of datasets keyed by condition:
        'simple', 'complex', 'both', 'complex_control', 'both_control'
    each containing X_train, y_train, X_test, y_test.
    """
    rng = np.random.RandomState(seed)

    # 1. Balanced binary target y in {0, 1}
    y = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # 2. Feature A: Simple 2D Linear Feature
    # Dimension 0: Informative linear signal (class 0: -1.0, class 1: +1.0)
    # Dimension 1: Independent uninformative Gaussian noise
    linear_signal = (2 * y - 1) * 1.0 + rng.normal(0.0, 0.35, size=n_samples)
    noise_A = rng.normal(0.0, 0.35, size=n_samples)
    feat_A = np.column_stack([linear_signal, noise_A])

    # 3. Feature B: Complex 2D Concentric Circles (Raw Non-linear decision boundary)
    # Inner ring (y=0, radius ~1.0), Outer ring (y=1, radius ~2.5)
    angles = rng.uniform(0.0, 2 * np.pi, size=n_samples)
    radii = np.where(y == 0, 1.0, 2.5) + rng.normal(0.0, 0.15, size=n_samples)
    feat_B = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # 4. Train / Test Split Indices (Shared across all features)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    split_idx = int(n_samples * (1 - test_split))
    tr, te = indices[:split_idx], indices[split_idx:]

    y_train, y_test = y[tr], y[te]

    # 5. Diagnostic Optimization Control: Feature B_control (2D)
    # Dimension 0: Normalized radial signal derived from Feature B.
    # Mean and standard deviation are computed strictly on training data to prevent leakage.
    # Dimension 1: Independent uninformative Gaussian noise.
    r = np.sqrt(feat_B[:, 0] ** 2 + feat_B[:, 1] ** 2)
    mu_r = np.mean(r[tr])
    std_r = np.std(r[tr])

    noise_ctrl = rng.normal(0.0, 0.35, size=n_samples)
    feat_B_ctrl_tr = np.column_stack([(r[tr] - mu_r) / std_r, noise_ctrl[tr]])
    feat_B_ctrl_te = np.column_stack([(r[te] - mu_r) / std_r, noise_ctrl[te]])

    datasets = {
        "simple": {
            "X_train": feat_A[tr],
            "X_test": feat_A[te],
            "y_train": y_train,
            "y_test": y_test,
        },
        "complex": {
            "X_train": feat_B[tr],
            "X_test": feat_B[te],
            "y_train": y_train,
            "y_test": y_test,
        },
        "both": {
            "X_train": np.column_stack([feat_A[tr], feat_B[tr]]),
            "X_test": np.column_stack([feat_A[te], feat_B[te]]),
            "y_train": y_train,
            "y_test": y_test,
        },
        "complex_control": {
            "X_train": feat_B_ctrl_tr,
            "X_test": feat_B_ctrl_te,
            "y_train": y_train,
            "y_test": y_test,
        },
        "both_control": {
            "X_train": np.column_stack([feat_A[tr], feat_B_ctrl_tr]),
            "X_test": np.column_stack([feat_A[te], feat_B_ctrl_te]),
            "y_train": y_train,
            "y_test": y_test,
        },
    }

    return datasets
