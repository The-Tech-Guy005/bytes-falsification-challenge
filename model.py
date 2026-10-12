"""
model.py - Standard, Beginner-Friendly Neural Network

Uses scikit-learn's standard MLPClassifier:
Input -> One Hidden Layer (16 neurons, ReLU) -> Output (Logistic / Sigmoid).
"""

from sklearn.neural_network import MLPClassifier


def build_model(seed=42):
    """
    Creates a small, reproducible 2-layer Neural Network.

    - hidden_layer_sizes=(16,): Exactly one hidden layer with 16 neurons.
    - activation='relu': Non-linear ReLU activation for hidden neurons.
    - solver='adam': Standard, robust gradient-based optimizer.
    - max_iter=500: Sufficient epochs for convergence without overfitting.
    - random_state=seed: Guarantees reproducible weight initialization.
    """
    return MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed,
    )
