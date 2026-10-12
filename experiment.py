"""
experiment.py - Multi-Seed Falsification Experiment Runner with Convergence Tracking

Addresses:
1. Multi-seed evaluation (20 independent random seeds).
2. Matched information structure (Feature A & Feature B_control both have [1 signal, 1 noise]).
3. Convergence metrics (n_iter, reached_max_iter, average learning curves).
4. Diagnostic Representation Control (Feature B vs Feature B_control).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import generate_dataset
from model import build_model


def run_single_seed(seed):
    """
    Runs all 5 conditions on a single random seed and returns metrics and loss curves.
    """
    rng = np.random.RandomState(seed)
    datasets = generate_dataset(n_samples=1000, seed=seed)

    condition_keys = ["simple", "complex", "both", "complex_control", "both_control"]
    metrics = {}
    loss_curves = {}

    for cond in condition_keys:
        d = datasets[cond]
        model = build_model(seed=seed)
        model.fit(d["X_train"], d["y_train"])

        train_acc = float(model.score(d["X_train"], d["y_train"]) * 100.0)
        test_acc = float(model.score(d["X_test"], d["y_test"]) * 100.0)
        n_iter = int(model.n_iter_)
        reached_max = int(n_iter >= 500)

        loss_curves[cond] = list(model.loss_curve_)

        record = {
            "seed": seed,
            "condition": cond,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_iter": n_iter,
            "reached_max_iter": reached_max,
            "pfi_drop_simple": np.nan,
            "pfi_drop_complex": np.nan,
        }

        # Permutation Feature Importance on multi-feature conditions
        if cond in ("both", "both_control"):
            X_test = d["X_test"].copy()
            y_test = d["y_test"]

            # Scramble Feature A (Columns 0 & 1)
            X_perm_A = X_test.copy()
            X_perm_A[:, :2] = rng.permutation(X_perm_A[:, :2])
            drop_A = test_acc - float(model.score(X_perm_A, y_test) * 100.0)

            # Scramble Feature B / B_control (Columns 2 & 3)
            X_perm_B = X_test.copy()
            X_perm_B[:, 2:] = rng.permutation(X_perm_B[:, 2:])
            drop_B = test_acc - float(model.score(X_perm_B, y_test) * 100.0)

            record["pfi_drop_simple"] = float(drop_A)
            record["pfi_drop_complex"] = float(drop_B)

        metrics[cond] = record

    return metrics, loss_curves


def run_experiment(num_seeds=20):
    os.makedirs("results", exist_ok=True)
    seeds = [100 + i for i in range(num_seeds)]

    print("=" * 75)
    print(f"  FALSIFICATION EXPERIMENT: SIMPLICITY BIAS ACROSS {num_seeds} SEEDS")
    print("=" * 75)
    print(f"Executing seeds: {seeds[0]} to {seeds[-1]}...\n")

    all_records = []
    condition_losses = {
        "simple": [],
        "complex": [],
        "both": [],
        "complex_control": [],
        "both_control": [],
    }

    for idx, s in enumerate(seeds):
        seed_metrics, seed_losses = run_single_seed(s)
        for cond, rec in seed_metrics.items():
            all_records.append(rec)
            condition_losses[cond].append(seed_losses[cond])
        if (idx + 1) % 5 == 0 or (idx + 1) == num_seeds:
            print(f"Completed seed {idx + 1}/{num_seeds} (seed={s})")

    # 1. Save Per-Seed Results
    df_per_seed = pd.DataFrame(all_records)
    df_per_seed.to_csv("results/per_seed_results.csv", index=False)
    print("\nSaved per-seed metrics to 'results/per_seed_results.csv'")

    # 2. Compute Summary Statistics (Mean, Std, Median, Min, Max, Convergence)
    summary_rows = []
    condition_display_names = {
        "simple": "1. Simple Only (2D Linear [1 signal, 1 noise])",
        "complex": "2. Complex Only (2D Raw Rings)",
        "both": "3. Both Features (Raw Competition)",
        "complex_control": "4. Diagnostic Control: Complex Only (2D Linearized [1 signal, 1 noise])",
        "both_control": "5. Diagnostic Control: Both Features (Control Competition)",
    }

    for cond, display_name in condition_display_names.items():
        sub = df_per_seed[df_per_seed["condition"] == cond]
        row = {
            "Condition": display_name,
            "Test_Acc_Mean": sub["test_acc"].mean(),
            "Test_Acc_Std": sub["test_acc"].std(),
            "Test_Acc_Median": sub["test_acc"].median(),
            "Test_Acc_Min": sub["test_acc"].min(),
            "Test_Acc_Max": sub["test_acc"].max(),
            "Drop_Simple_Mean": sub["pfi_drop_simple"].mean(),
            "Drop_Simple_Std": sub["pfi_drop_simple"].std(),
            "Drop_Complex_Mean": sub["pfi_drop_complex"].mean(),
            "Drop_Complex_Std": sub["pfi_drop_complex"].std(),
            "Iterations_Mean": sub["n_iter"].mean(),
            "Iterations_Std": sub["n_iter"].std(),
            "Iterations_Max": sub["n_iter"].max(),
            "Num_Runs_Max_Iter": int(sub["reached_max_iter"].sum()),
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/summary_metrics.csv", index=False)
    print("Saved aggregate summary to 'results/summary_metrics.csv'")

    # Print clean summary table
    print("\n" + "=" * 75)
    print("AGGREGATE EXPERIMENTAL SUMMARY (20 SEEDS)")
    print("=" * 75)
    for _, r in df_summary.iterrows():
        print(f"\n{r['Condition']}:")
        print(f"  Test Accuracy : {r['Test_Acc_Mean']:.2f}% ± {r['Test_Acc_Std']:.2f}% (Median: {r['Test_Acc_Median']:.2f}%)")
        print(f"  Iterations    : {r['Iterations_Mean']:.1f} ± {r['Iterations_Std']:.1f} (Max: {r['Iterations_Max']}, Reached 500 iter: {r['Num_Runs_Max_Iter']}/{num_seeds})")
        if not np.isnan(r["Drop_Simple_Mean"]):
            print(f"  Drop Simple   : {r['Drop_Simple_Mean']:.2f}% ± {r['Drop_Simple_Std']:.2f}%")
            print(f"  Drop Complex  : {r['Drop_Complex_Mean']:.2f}% ± {r['Drop_Complex_Std']:.2f}%")
    print("=" * 75 + "\n")

    # 3. Process Learning Curves
    max_len = max(max(len(c) for c in curves) for curves in condition_losses.values())
    avg_loss_curves = {}
    for cond, curves in condition_losses.items():
        padded = []
        for c in curves:
            pad = c + [c[-1]] * (max_len - len(c))
            padded.append(pad)
        avg_loss_curves[cond] = np.mean(padded, axis=0)

    df_learning = pd.DataFrame(avg_loss_curves)
    df_learning.to_csv("results/learning_curves.csv", index_label="Iteration")
    print("Saved average learning curves to 'results/learning_curves.csv'")

    # 4. Generate Publication-Style Diagnostic Figures
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Falsification Study: Simplicity Preference vs. Optimization Geometry (20 Seeds)",
        fontsize=13,
        fontweight="bold",
    )

    # Plot 1: Optimization Dynamics / Learning Curves
    ax1 = axes[0, 0]
    ax1.plot(avg_loss_curves["simple"], label="Simple Only (Linear [1 sig, 1 noise])", color="#2b5c8f", lw=2)
    ax1.plot(avg_loss_curves["complex"], label="Complex Only (Raw Rings 2D)", color="#d95f02", lw=2)
    ax1.plot(avg_loss_curves["complex_control"], label="Control: Complex (Linearized [1 sig, 1 noise])", color="#7570b3", lw=2, linestyle="--")
    ax1.plot(avg_loss_curves["both"], label="Both Features (Raw Competition)", color="#1b9e77", lw=2)
    ax1.set_title("1. Training Loss Convergence (20 Seeds)", fontweight="bold")
    ax1.set_xlabel("Optimization Iteration")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Iterations to Convergence Across Conditions
    ax2 = axes[0, 1]
    short_labels = ["Simple", "Complex\n(Raw)", "Both\n(Raw)", "Ctrl Complex\n(Linearized)", "Both Ctrl"]
    means_iter = df_summary["Iterations_Mean"]
    stds_iter = df_summary["Iterations_Std"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    bars_iter = ax2.bar(short_labels, means_iter, yerr=stds_iter, capsize=4, color=colors, width=0.5)
    ax2.set_title("2. Optimization Speed (Iterations to Converge)", fontweight="bold")
    ax2.set_ylabel("Mean Iterations")
    for bar in bars_iter:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:.0f}", ha="center", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 3: Feature Reliance in Main Raw Competition (Condition 3)
    ax3 = axes[1, 0]
    row_both = df_summary[df_summary["Condition"].str.contains("Raw Competition")].iloc[0]
    drops_raw = [row_both["Drop_Simple_Mean"], row_both["Drop_Complex_Mean"]]
    errs_raw = [row_both["Drop_Simple_Std"], row_both["Drop_Complex_Std"]]
    b3 = ax3.bar(["Simple A\n(2D Linear)", "Complex B\n(2D Raw Rings)"], drops_raw, yerr=errs_raw, capsize=5, color=["#2b5c8f", "#d95f02"], width=0.45)
    ax3.set_title("3. Reliance in Raw Competition (Condition 3)", fontweight="bold")
    ax3.set_ylabel("Accuracy Drop When Corrupted (%)")
    for b in b3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, max(0.5, h + 1.5), f"{h:.1f}%", ha="center", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 4: Feature Reliance in Diagnostic Control Competition (Condition 5)
    ax4 = axes[1, 1]
    row_ctrl = df_summary[df_summary["Condition"].str.contains("Control Competition")].iloc[0]
    drops_ctrl = [row_ctrl["Drop_Simple_Mean"], row_ctrl["Drop_Complex_Mean"]]
    errs_ctrl = [row_ctrl["Drop_Simple_Std"], row_ctrl["Drop_Complex_Std"]]
    b4 = ax4.bar(["Simple A\n(2D Linear)", "Control B\n(2D Linearized)"], drops_ctrl, yerr=errs_ctrl, capsize=5, color=["#2b5c8f", "#7570b3"], width=0.45)
    ax4.set_title("4. Reliance in Control Competition (Condition 5)", fontweight="bold")
    ax4.set_ylabel("Accuracy Drop When Corrupted (%)")
    for b in b4:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2, max(0.5, h + 1.5), f"{h:.1f}%", ha="center", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = "results/experiment_results.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved diagnostic plots to '{plot_path}'\n")


if __name__ == "__main__":
    run_experiment(num_seeds=20)
