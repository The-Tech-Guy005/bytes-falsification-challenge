# Research Proposal & Experimental Defense
## Investigating Simplicity Bias in Neural Networks: A Controlled Falsification Study

**Author:** Second-Year B.Tech Computer Science & Engineering Student  
**Project Task:** BYTES ML Research Challenge — The Falsification Challenge  
**Repository Working Directory:** `/Users/hridayarora/Documents/BYTES TASK 1 ML R`  
**Primary Deliverable Files:** `data.py`, `model.py`, `experiment.py`, `app.py`, `README.md`, `results/`  

---

## 1. Executive Summary & Research Context

A prominent claim in recent deep learning literature (e.g., Shah et al., 2020) is that neural networks exhibit an inherent **"simplicity bias"**—an inductive inclination to rely almost exclusively on simpler, lower-complexity predictive features when multiple predictive signals are simultaneously available, even when more complex features possess equal or greater predictive capacity.

In scientific research, however, demonstrating that a network uses a simpler feature does not inherently prove that the network possesses a fundamental preference for simplicity. The goal of this research project is **falsification-oriented experimentation**:
1. Formulate the simplicity bias claim as a testable empirical hypothesis.
2. Formulate plausible alternative explanations—specifically, **representation geometry** and **optimization difficulty**.
3. Construct a strictly controlled synthetic benchmark where feature dimensionality, information structure, noise levels, and isolated predictive capacity are carefully matched.
4. Introduce diagnostic representation controls and multi-seed convergence profiling across 20 independent random seeds.
5. Evaluate whether the observed feature reliance reflects an immutable preference for simple semantic concepts or is instead substantially driven by gradient and optimization dynamics.

---

## 2. Core Research Question

> **Primary Research Question:**  
> *When a shallow neural network has access to multiple predictive features of differing structural complexity, does gradient-based optimization with Adam preferentially rely on the simpler predictive feature, and to what extent is the observed reliance influenced by representation geometry and optimization difficulty?*

---

## 3. Hypotheses & Theoretical Framework

### 3.1 Original Simplicity-Bias Hypothesis ($H_1$)
When presented with both a simple feature (linearly separable) and a complex feature (non-linearly separable) that individually offer comparable isolated predictive accuracy, gradient-based optimization will preferentially rely on the simple feature due to an inductive bias toward simpler hypothesis classes.

### 3.2 Null Hypothesis ($H_0$)
The neural network will not prioritize the simpler feature: attribution will be distributed equally between features, or will vary stochastically according to random parameter initialization without regard to structural simplicity.

### 3.3 The Alternative Explanation ($H_{\text{alt}}$: Representation & Optimization Bottleneck)
The apparent preference for the linear feature is not necessarily caused by an intrinsic bias toward simplicity itself. Instead:
- In raw Cartesian coordinates $(u_1, u_2)$, a concentric non-linear decision boundary creates a curved, non-convex loss surface that requires multiple ReLU hidden units to coordinate piecewise linear partitions.
- Conversely, a linear feature $(x_1, x_2)$ presents an immediate, monotonic gradient slope.
- The optimizer reduces loss much faster along the linear dimension. Once the loss approaches zero, backpropagated error gradients $(\hat{y} - y)$ vanish. This starves the complex feature of gradient updates before its non-linear partition can form.
- Under this alternative hypothesis, the observed reliance is an artifact of **optimization speed and representation geometry**, rather than an immutable preference for simpler concepts.

---

## 4. Synthetic Dataset & Information-Structure Design

To eliminate confounding variables present in observational data (such as hidden correlations, unequal feature dimensions, or unmeasured noise), we design a synthetic dataset ($N = 1000$ samples, balanced binary targets $y \in \{0, 1\}$, 70% train / 30% test split).

### 4.1 Feature Construction & Information Matching

A common flaw in prior prototypes was confounding dimensionality (e.g., comparing a 1D linear feature directly to a 2D circle). To ensure rigorous parity, **every individual feature in this study is strictly 2D** and constructed with a matched information structure:

1. **Feature A (Simple 2D Linear Feature):**
   - **Dimension 0 (Informative Linear Signal):**
     $$x_{A1} = (2y - 1) \cdot 1.0 + \epsilon_1, \quad \epsilon_1 \sim \mathcal{N}(0, 0.35^2)$$
     Class 0 centers at $-1.0$; Class 1 centers at $+1.0$. A single linear threshold at 0 separates classes.
   - **Dimension 1 (Independent Noise):**
     $$x_{A2} = \epsilon_2, \quad \epsilon_2 \sim \mathcal{N}(0, 0.35^2)$$
   - Structure: `[1 informative signal, 1 independent noise]`.

2. **Feature B (Complex 2D Non-Linear Feature — Concentric Rings):**
   - Constructed in Cartesian coordinates from polar distributions:
     $$\theta \sim \text{Uniform}(0, 2\pi)$$
     $$r = \begin{cases} 1.0 + \mathcal{N}(0, 0.15^2) & \text{if } y = 0 \\ 2.5 + \mathcal{N}(0, 0.15^2) & \text{if } y = 1 \end{cases}$$
     $$u_{B1} = r \cos(\theta), \quad u_{B2} = r \sin(\theta)$$
   - Linearly inseparable: a linear classifier achieves ~50% (random guess), while a 2-layer ReLU MLP can learn the circular boundary.
   - Structure: `[2D coupled non-linear coordinate representation]`.

3. **Feature B_control (Diagnostic Control Feature — Linearized Radial Representation):**
   - Extracts the exact underlying radial signal from Feature B:
     $$r = \sqrt{u_{B1}^2 + u_{B2}^2}$$
   - Normalized using training-set statistics only to prevent data leakage:
     $$r_{\text{norm}} = \frac{r - \mu_{r,\text{train}}}{\sigma_{r,\text{train}}}$$
   - Paired with an independent noise dimension ($\sigma = 0.35$):
     $$\text{feat\_B\_ctrl} = [r_{\text{norm}}, \epsilon_{\text{ctrl}}], \quad \epsilon_{\text{ctrl}} \sim \mathcal{N}(0, 0.35^2)$$
   - Structure: `[1 informative radial signal, 1 independent noise]`.
   - **Diagnostic Purpose:** Feature B_control retains 100% of Feature B's underlying predictive information, but eliminates the coordinate curvature bottleneck.

---

## 5. The Five Experimental Conditions

To disentangle simplicity bias from optimization difficulty, the experiment implements five specific conditions:

| Condition | Inputs | Input Dim | Research Function |
| :--- | :--- | :---: | :--- |
| **1. Simple Only** | Feature A | 2D | Baseline: Confirms linear feature is easily learned. |
| **2. Complex Only** | Raw Feature B | 2D | Baseline: Confirms MLP has capacity to learn raw concentric rings alone. |
| **3. Both Features (Raw Competition)** | Feature A + Raw Feature B | 4D | **Core Competition Test:** Evaluates reliance when both features compete directly. |
| **4. Complex Control Only** | Feature B_control | 2D | Diagnostic Baseline: Confirms linearized radial feature is learnable. |
| **5. Both Features (Control Competition)** | Feature A + Feature B_control | 4D | **Diagnostic Falsification Test:** Evaluates reliance when both features have matched `[1 signal, 1 noise]` linear accessibility. |

---

## 6. Model Architecture & Training Protocol

To keep the project transparent, explainable, and free of hidden framework abstractions, the model is implemented using `scikit-learn`'s standard multi-layer perceptron:

- **Model Class:** `sklearn.neural_network.MLPClassifier`
- **Topology:**
  - Input Layer ($d \in \{2, 4\}$ depending on condition)
  - Hidden Layer: Exactly **16 neurons** with **ReLU** activation ($f(z) = \max(0, z)$)
  - Output Layer: **1 neuron** with **Logistic/Sigmoid** activation ($\sigma(z) = \frac{1}{1 + e^{-z}}$)
- **Loss Function:** Binary Cross-Entropy (Log-loss)
- **Optimizer:** Adam ($\alpha = 0.001$, standard adaptive momentum)
- **Iteration Ceiling:** `max_iter = 500`
- **Training/Testing Split:** Stratified 70% train ($N=700$), 30% test ($N=300$)
- **Data Leakage Safeguard:** Normalization statistics for the diagnostic control ($\mu_r, \sigma_r$) are calculated strictly on the training partition and applied to the test partition.

---

## 7. Multi-Seed Robustness & Evaluation Methodology

### 7.1 Multi-Seed Protocol (20 Independent Seeds)
Single-seed evaluations suffer from the "initialization lottery" (Fort et al., 2020), where stochastic weight initialization can randomly favor one feature. To guarantee reproducibility and statistical stability:
- The entire 5-condition protocol is executed across **20 deterministic seeds** (`seeds = [100, 101, ..., 119]`).
- For every condition and seed, data generation, train/test partitioning, model initialization, and training are conducted independently.
- We report: **Mean**, **Standard Deviation**, **Median**, **Minimum**, and **Maximum**.

### 7.2 Permutation Feature Importance (PFI)
To measure which feature the network actually relies upon in multi-feature conditions (Conditions 3 and 5), we use Permutation Feature Importance:
1. Measure baseline classification accuracy on the clean test set: $\text{Acc}_{\text{base}}$.
2. Permute/shuffle Feature Group A (columns 0 & 1) across test instances, breaking its correlation with the target while maintaining its marginal distribution. Measure new accuracy $\text{Acc}_{\text{perm\_A}}$. Compute accuracy drop:
   $$\Delta \text{Acc}_A = \text{Acc}_{\text{base}} - \text{Acc}_{\text{perm\_A}}$$
3. Restore Feature Group A. Permute Feature Group B (columns 2 & 3) across test instances. Compute accuracy drop:
   $$\Delta \text{Acc}_B = \text{Acc}_{\text{base}} - \text{Acc}_{\text{perm\_B}}$$
4. An accuracy collapse indicates that the network's predictions depend heavily on that feature group.

### 7.3 Convergence Profiling
For each condition and seed, we log:
- Total iterations required to converge (`model.n_iter_`).
- Number of runs hitting the `max_iter = 500` threshold.
- Epoch-by-epoch loss trajectories saved in `results/learning_curves.csv`.

---

## 8. Empirical Findings (Actual CSV Data Across 20 Seeds)

Below are the exact values recorded in `results/summary_metrics.csv` generated by `experiment.py`:

```text
=================================================================================================================================
AGGREGATE EXPERIMENTAL SUMMARY (20 INDEPENDENT DETERMINISTIC SEEDS: 100 TO 119)
=================================================================================================================================
Condition                                           Test Acc (Mean ± Std)    Median    Mean Iterations (Max 500)   PFI Drop Simple   PFI Drop Complex
1. Simple Only (2D Linear [1 sig, 1 noise])                99.78% ± 0.25%    99.83%         288.1 ± 31.1 (0/20)         N/A               N/A
2. Complex Only (2D Raw Rings)                            100.00% ± 0.00%   100.00%         466.9 ± 27.1 (6/20)         N/A               N/A
3. Both Features (Raw Competition)                         99.97% ± 0.10%   100.00%         274.4 ± 29.4 (0/20)    49.15% ± 2.90%    0.72% ± 0.60%
4. Diagnostic Control: Complex Only (Linearized)          100.00% ± 0.00%   100.00%         262.9 ± 26.8 (0/20)         N/A               N/A
5. Diagnostic Control: Both Features (Control)            100.00% ± 0.00%   100.00%         210.0 ± 22.6 (0/20)    22.02% ± 12.86%  27.92% ± 12.63%
=================================================================================================================================
```

### 8.1 Key Observations from the Empirical Data

1. **Isolated Baselines Confirm Equal High Accuracy (Conditions 1, 2, and 4):**
   - Simple Feature A alone achieves **$99.78\% \pm 0.25\%$** test accuracy.
   - Raw Complex Feature B alone achieves **$100.00\% \pm 0.00\%$** test accuracy.
   - Linearized Complex Control B_control alone achieves **$100.00\% \pm 0.00\%$** test accuracy.
   - This empirically confirms that the 16-neuron ReLU MLP has more than enough expressive capacity to learn the non-linear concentric circle boundary on its own.

2. **The Raw Competition Result (Condition 3):**
   - When Feature A and Raw Feature B are presented together, the model achieves **$99.97\% \pm 0.10\%$** test accuracy.
   - Scrambling Simple Feature A causes test accuracy to drop by **$49.15\% \pm 2.90\%$** (collapsing model performance to ~50.8%, virtually random guessing on balanced binary data).
   - Scrambling Raw Complex Feature B causes an accuracy drop of only **$0.72\% \pm 0.60\%$**.
   - **Superficial Interpretation:** If this were the only experiment conducted, one would conclude that the model exhibits an overwhelming simplicity bias.

3. **The Diagnostic Control Result (Condition 5):**
   - When the non-linear signal is presented in the linearized representation (`[1 signal, 1 noise]`), **the reliance on Simple Feature A drops from $49.15\%$ down to $22.02\% \pm 12.86\%$**.
   - Reliance on the radial feature surges from **$0.72\%$ up to $27.92\% \pm 12.63\%$**.
   - In this control setting, the model distributes attribution across both features roughly equally (with a slight edge toward the radial signal).

4. **Optimization Convergence Profiling:**
   - **Condition 1 (Simple Only):** Converges in **$288.1 \pm 31.1$ iterations**; 0 of 20 runs reach 500 iterations.
   - **Condition 2 (Raw Complex Only):** Converges in **$466.9 \pm 27.1$ iterations**; **6 of 20 runs hit the 500-iteration ceiling**.
   - **Condition 4 (Linearized Control Only):** Converges in **$262.9 \pm 26.8$ iterations**; 0 of 20 runs reach 500 iterations.
   - **Learning Curves:** As recorded in `results/learning_curves.csv`, loss along the linear dimension drops below $0.40$ in fewer than 20 iterations, whereas the raw concentric boundary requires dozens of iterations to establish initial partitions.

---

## 9. Critical Analysis: What the Evidence Supports vs. What It Does Not Establish

### 9.1 What the Evidence Supports
1. In raw Cartesian coordinates, shallow neural networks trained with Adam preferentially rely on linearly separable features over concentric non-linear features when both are present.
2. The diagnostic control provides clear evidence that **representation geometry and optimization difficulty substantially influence this observed reliance**.
3. When the non-linear signal is provided in an optimization-accessible coordinate system, the overwhelming preference for the linear feature dissolves into balanced feature usage.
4. The convergence measurements confirm that the raw non-linear feature is markedly slower to optimize than both the linear feature and the linearized radial feature.

### 9.2 What the Evidence Does NOT Establish
1. **No Universal Law of Simplicity Bias:** The experiment does not prove that neural networks possess an inherent, universal drive toward semantic simplicity across arbitrary architectures and domains.
2. **Comparable Accuracy Does Not Mean Equal Signal Strength:** Although all individual conditions reach ~100% test accuracy, classification accuracy is a thresholded step function. It does not prove that the underlying margin distributions, mutual information, or Fisher information are strictly identical between features.
3. **Diagnostic Intervention vs. Causal Proof:** The engineered radial transformation alters the input geometry. While this proves that coordinate geometry governs feature reliance, it serves as a diagnostic demonstration rather than a fully decoupled mathematical proof of loss surface dynamics.
4. **Permutation Importance Measures Input Reliance, Not Internal Mechanism:** PFI measures test-time output sensitivity to input scrambling; it does not directly observe the weight trajectory through parameter space during training.

---

## 10. Confounders & Methodological Limitations

1. **Ceiling Effect in Classification Accuracy:** All isolated features achieve test accuracy between 99.78% and 100.00%. Because accuracy saturates near 100%, subtle differences in signal strength cannot be detected by classification accuracy alone.
2. **Low-Dimensional Synthetic Scope:** The experiment is conducted on 2D and 4D synthetic distributions. In modern deep learning (e.g., Vision Transformers or deep ResNets on ImageNet), simplicity bias occurs across thousands of high-dimensional, entangled feature manifolds where linearizations cannot be manually engineered.
3. **Architecture and Optimizer Specifics:** The experiment uses a single hidden layer of 16 ReLU units trained with Adam. The dynamics may differ under unaccelerated stochastic gradient descent (SGD), different learning rates, weight decay ($L_2$ regularization), or deeper architectures.

---

## 11. Final Conclusion

> **Final Conclusion:**  
> In this controlled synthetic benchmark, a shallow neural network trained with Adam exhibits a pronounced empirical reliance on the simple linear feature when competing against a raw non-linear concentric ring feature.  
> However, our diagnostic representation controls and convergence metrics demonstrate that **this observed preference is substantially influenced by representation geometry and optimization difficulty**, rather than reflecting an immutable preference for simpler features.  
> When the non-linear signal is supplied in an optimization-accessible coordinate representation, the network utilizes both features comparably. Therefore, claims of "simplicity bias" in shallow networks must be carefully evaluated against the alternative explanation of gradient and optimization bottlenecks.

---

## 12. Project Structure & Verification Guide

```text
BYTES TASK 1 ML R/
├── data.py                 # Clean synthetic data generator with matched [1 signal, 1 noise] structure
├── model.py                # 2-layer MLPClassifier (16 ReLU hidden units, Adam optimizer)
├── experiment.py           # 20-seed experiment runner, PFI calculator, and convergence logger
├── app.py                  # Streamlit demonstration interface for live inspection
├── requirements.txt        # Dependencies: scikit-learn, pandas, numpy, matplotlib, streamlit
├── README.md               # User-facing summary and execution instructions
├── research_proposal.md    # This complete research proposal and experimental defense
└── results/
    ├── summary_metrics.csv      # Aggregate statistics across all 20 seeds
    ├── per_seed_results.csv     # Granular seed-by-seed experimental logs
    ├── learning_curves.csv      # Iteration-by-iteration loss trajectories
    └── experiment_results.png   # 4-panel diagnostic visualization
```

### Reproducibility Commands
To verify the complete 20-seed pipeline from scratch:
```bash
python3 experiment.py
```
To launch the interactive demonstration panel:
```bash
streamlit run app.py
```
