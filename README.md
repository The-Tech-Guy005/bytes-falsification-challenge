# The Falsification Challenge: Investigating Simplicity Bias in Neural Networks
### BYTES ML Research Task — Multi-Seed Falsification Study

---

## 1. Research Question
*When a shallow neural network has access to multiple predictive features of differing structural complexity, does gradient-based optimization with Adam preferentially rely on the simpler predictive feature, and to what extent is the observed reliance influenced by representation geometry and optimization difficulty?*

---

## 2. Hypotheses & Theoretical Framework

- **Original Hypothesis ($H_1$)**: When presented with both a simple (2D linear) feature and a complex (2D non-linear concentric circle) feature of comparable isolated predictive accuracy, gradient-based optimization with Adam will preferentially rely on the simpler feature.
- **Null Hypothesis ($H_0$)**: The neural network will allocate reliance equally between both features or rely on features independent of structural simplicity.
- **The Optimization / Representation Alternative ($H_{\text{alt}}$)**:
  - The raw non-linear concentric circle feature in 2D Cartesian coordinates $(u_1, u_2)$ creates a non-convex, curved loss surface requiring hidden neurons to coordinate piecewise linear segments.
  - In contrast, the linear feature $(x_1, x_2)$ presents an immediate, monotonic gradient slope.
  - Slower loss reduction on the complex feature may result in reduced gradient updates once the linear feature lowers the loss, producing an apparent preference that is driven by optimization dynamics rather than an inherent avoidance of non-linear information.

---

## 3. Experimental Design & Methodological Controls

1. **Information-Structure & Dimensionality Matching**:
   - **Feature A (Simple 2D Linear)**: Dimension 0 contains 1D linear signal ($(2y-1) \cdot 1.0 + \text{noise}$); Dimension 1 contains independent uninformative Gaussian noise ($\sigma = 0.35$).
   - **Feature B (Complex 2D Raw Concentric Rings)**: 2D non-linear concentric rings in Cartesian coordinates ($r_{\text{inner}}=1.0, r_{\text{outer}}=2.5$).
   - **Feature B_control (Diagnostic Control 2D)**: Dimension 0 contains normalized radial signal derived from Feature B ($r = \sqrt{u_1^2 + u_2^2}$); Dimension 1 contains independent uninformative Gaussian noise.
   - *Data Leakage Prevention*: Normalization mean ($\mu_r$) and standard deviation ($\sigma_r$) for Feature B_control are computed strictly on training samples and applied to test samples.
2. **Sample & Split Controls**:
   - $N = 1000$ samples (700 train / 300 test), 50/50 balanced classes, shared train/test split indices across all conditions.
3. **Multi-Seed Robustness (20 Independent Seeds)**:
   - All 5 conditions are executed across **20 deterministic random seeds** (seeds 100 to 119) to mitigate initialization sensitivity.
4. **Role of Diagnostic Control**:
   - Feature B_control is a **diagnostic intervention**, not a perfectly matched natural control. It tests whether transforming the non-linear signal into an optimization-accessible coordinate alters the model's feature reliance.

---

## 4. Experimental Conditions

- **Condition 1 (Simple Only)**: 2D Linear feature `[1 signal, 1 noise]`.
- **Condition 2 (Complex Only)**: 2D Raw concentric rings.
- **Condition 3 (Both Features - Raw Competition)**: 4D input (2D Simple + 2D Raw Complex).
- **Condition 4 (Diagnostic Control - Complex Only)**: 2D Linearized radial feature `[1 signal, 1 noise]`.
- **Condition 5 (Diagnostic Control - Both Features)**: 4D input (2D Simple + 2D Linearized Complex Control).

---

## 5. Model Architecture & Hyperparameters

- **Model**: `sklearn.neural_network.MLPClassifier`
  - Architecture: Input $\to$ Hidden Layer (16 neurons, **ReLU**) $\to$ Output (Logistic/Sigmoid).
  - Optimizer: Adam (`max_iter=500`, default $\alpha=0.001$).
  - Loss: Binary Cross-Entropy (Log-loss).
- **Evaluation Metric**: Permutation Feature Importance (PFI) — test accuracy drop ($\Delta \text{Acc}$) when Feature Group A vs Feature Group B is permuted across test rows.

---

## 6. Actual Experimental Results (Aggregated Across 20 Seeds)

```text
===========================================================================================================================
AGGREGATE EXPERIMENTAL SUMMARY (20 INDEPENDENT SEEDS)
===========================================================================================================================
Condition                                       Test Acc (Mean ± Std)   Median    Mean Iterations (Max 500)   PFI Drop Simple   PFI Drop Complex
1. Simple Only (2D Linear [1 sig, 1 noise])            99.78% ± 0.25%   99.83%          288.1 ± 31.1 (0/20)         N/A               N/A
2. Complex Only (2D Raw Rings)                        100.00% ± 0.00%  100.00%          466.9 ± 27.1 (6/20)         N/A               N/A
3. Both Features (Raw Competition)                     99.97% ± 0.10%  100.00%          274.4 ± 29.4 (0/20)    49.15% ± 2.90%    0.72% ± 0.60%
4. Diagnostic Control: Complex Only (Linearized)      100.00% ± 0.00%  100.00%          262.9 ± 26.8 (0/20)         N/A               N/A
5. Diagnostic Control: Both Features (Control)        100.00% ± 0.00%  100.00%          210.0 ± 22.6 (0/20)    22.02% ± 12.86%  27.92% ± 12.63%
===========================================================================================================================
```

*Detailed per-seed metrics: `results/per_seed_results.csv`*  
*Iteration loss curves: `results/learning_curves.csv`*  
*Diagnostic research plots: `results/experiment_results.png`*

---

## 7. Analysis: Observed Reliance vs. Optimization Difficulty

### A. Raw Competition (Condition 3):
- When the network is trained on both raw features, it relies predominantly on the simple linear feature:
  - Scrambling Feature A drops test accuracy by **49.15% ± 2.90%** (reducing the model to ~50.8% near-chance performance).
  - Scrambling Feature B drops test accuracy by only **0.72% ± 0.60%**.

### B. Diagnostic Control Competition (Condition 5):
- When the non-linear signal is presented in a linearized coordinate representation `[1 signal, 1 noise]`, reliance on the simple feature decreases from **49.15% to 22.02%**, while reliance on the radial feature increases from **0.72% to 27.92% ± 12.63%**.
- In this diagnostic setting, the model utilizes both features to a comparable degree.

### C. Optimization Convergence Metrics:
- **Condition 1 (Simple Only)** converged in **288.1 ± 31.1 iterations** (0 of 20 runs reached 500 iterations).
- **Condition 2 (Raw Complex Only)** required **466.9 ± 27.1 iterations**, with **6 of 20 runs reaching the 500-iteration ceiling**.
- **Condition 4 (Linearized Complex Control)** converged in **262.9 ± 26.8 iterations** (0 of 20 runs reached 500 iterations), matching the convergence pace of the simple feature.

---

## 8. Scientific Nuances & What the Results Do NOT Establish

1. **Representation and Optimization Strongly Influence Reliance**: The diagnostic control indicates that representation geometry and optimization difficulty substantially affect which feature the network relies upon.
2. **Comparable Isolated Accuracy $\neq$ Equal Signal Strength**: Even though isolated test accuracy is ~100% across all individual conditions, classification accuracy is a thresholded metric. It does not establish that the underlying margin distributions, mutual information, or Fisher information are strictly identical.
3. **Diagnostic Intervention vs. Causal Proof**: The radial transformation alters the input geometry. While this confirms that geometric representation affects feature selection, it serves as a diagnostic indicator rather than a complete causal proof of optimizer dynamics.
4. **PFI Measures Input Reliance**: Permutation importance measures sensitivity of the trained model to input scrambling at test time, but does not isolate the exact parameter-level trajectory during training.

---

## 9. Methodological Limitations

1. **Synthetic Scope**: The experiment evaluates low-dimensional 2D/4D synthetic distributions; findings do not directly generalize to high-dimensional hierarchical deep architectures.
2. **Architecture and Optimizer Specifics**: Evaluated on a 16-neuron single-hidden-layer MLP with the Adam optimizer; differing capacities, activations, regularization ($L_1 / L_2$), or optimizers (e.g., SGD) could exhibit different dynamics.

---

## 10. Final Conclusion

> **Conclusion**: In this controlled synthetic benchmark, the network exhibits a pronounced empirical preference for the simple linear feature when competing against the raw non-linear feature. However, the diagnostic control and convergence metrics provide evidence that **representation and optimization difficulty substantially influence this observed preference**. When the same non-linear signal is provided in an optimization-accessible representation, feature reliance becomes balanced; the results suggest that the raw effect is substantially influenced by representation and optimization dynamics, rather than providing evidence for an immutable preference for simpler features.

---

## 11. How to Run

```bash
# Execute the full 20-seed experiment and generate all CSVs & plots
python3 experiment.py

# Launch the interactive Streamlit demonstration interface
streamlit run app.py
```
