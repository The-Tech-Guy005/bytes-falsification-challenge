"""
app.py - Minimal Interactive Interface for Demonstration

Streamlit demonstration app for the Falsification Challenge:
- Inspect any of the 5 conditions on demand.
- View test accuracy, learning dynamics, and permutation drops.
- Explains the contrast between Raw Competition and Optimization Control.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data import generate_dataset
from model import build_model

st.set_page_config(page_title="Simplicity Bias Falsification", layout="centered")

st.title("The Falsification Challenge: Simplicity Bias")
st.write(
    "A research prototype investigating whether shallow neural networks genuinely prefer "
    "simple features, or if the preference is confounded by optimization difficulty."
)

condition_options = {
    "1. Simple Feature Only (2D Linear [1 signal, 1 noise])": "simple",
    "2. Complex Feature Only (2D Raw Concentric Rings)": "complex",
    "3. Both Features: Raw Competition (2D Simple + 2D Raw Complex)": "both",
    "4. Diagnostic Control: Complex Only (2D Linearized [1 signal, 1 noise])": "complex_control",
    "5. Diagnostic Control: Both Features (Simple + Complex Control)": "both_control",
}

choice = st.selectbox("Select Experimental Condition:", list(condition_options.keys()))
cond_key = condition_options[choice]

seed = st.number_input("Random Seed (for reproducibility):", min_value=1, max_value=999, value=42, step=1)

if st.button("Run Condition", type="primary"):
    with st.spinner("Training model..."):
        datasets = generate_dataset(n_samples=1000, seed=seed)
        d = datasets[cond_key]

        model = build_model(seed=seed)
        model.fit(d["X_train"], d["y_train"])

        train_acc = float(model.score(d["X_train"], d["y_train"]) * 100.0)
        test_acc = float(model.score(d["X_test"], d["y_test"]) * 100.0)

    st.success("Training Complete!")

    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", f"{train_acc:.2f}%")
    col2.metric("Testing Accuracy", f"{test_acc:.2f}%")

    st.subheader("Training Loss Curve")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(model.loss_curve_, color="#1f77b4", lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close()

    # Feature Reliance for multi-feature conditions
    if cond_key in ("both", "both_control"):
        st.subheader("Feature Reliance (Permutation Test)")
        rng = np.random.RandomState(seed)
        X_test = d["X_test"].copy()
        y_test = d["y_test"]

        # Permute Feature A (Columns 0 & 1)
        X_perm_A = X_test.copy()
        X_perm_A[:, :2] = rng.permutation(X_perm_A[:, :2])
        drop_A = test_acc - float(model.score(X_perm_A, y_test) * 100.0)

        # Permute Feature B / B_ctrl (Columns 2 & 3)
        X_perm_B = X_test.copy()
        X_perm_B[:, 2:] = rng.permutation(X_perm_B[:, 2:])
        drop_B = test_acc - float(model.score(X_perm_B, y_test) * 100.0)

        st.write(f"- Accuracy drop when **Simple Feature** is scrambled: **{drop_A:.2f}%**")
        feature_b_name = "Complex Feature" if cond_key == "both" else "Control Complex Feature"
        st.write(f"- Accuracy drop when **{feature_b_name}** is scrambled: **{drop_B:.2f}%**")

    # Research Interpretation
    st.subheader("Research Interpretation")
    if cond_key == "simple":
        st.info("The 2D linear feature converges quickly to 100% accuracy, confirming that simple hyperplanes provide steep, clean early gradients.")
    elif cond_key == "complex":
        st.info("The 2D non-linear circular boundary is also learnable to ~100% accuracy by the 16-neuron ReLU hidden layer, but requires significantly more iterations to converge.")
    elif cond_key == "both":
        st.warning("In raw competition, the network relies almost exclusively on the simple linear feature. However, whether this reflects intrinsic 'simplicity bias' or optimization speed differences is tested in Condition 5.")
    elif cond_key == "complex_control":
        st.info("When the non-linear radius signal is provided in a linearized coordinate representation, it converges much faster, matching the optimization speed of the linear feature.")
    elif cond_key == "both_control":
        st.success("Diagnostic Control: When the non-linear feature's optimization bottleneck is removed, we inspect whether the network still displays overwhelming reliance on the original linear feature, directly testing the optimization confounder.")
