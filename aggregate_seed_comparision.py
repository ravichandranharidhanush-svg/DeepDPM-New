"""
aggregate_seed_comparison.py

Aggregates baseline-vs-treatment DeepDPM comparisons across multiple seeds.
One favorable run doesn't tell you much on its own -- what matters is
whether the effect holds direction consistently across seeds, and whether
its size is larger than the seed-to-seed noise. This script:

    - parses each seed's baseline and treatment log (reusing parse_log /
      compute_convergence from compare_baseline_vs_contrastive.py)
    - reports mean +/- std of final NMI/ARI/ACC/K for baseline and
      treatment separately
    - reports the PAIRED per-seed difference (treatment - baseline) --
      this is the number that actually answers "does it help" -- along
      with a simple win-count ("treatment beat baseline on N/3 seeds")
    - reports convergence-speed stats (mean epoch to reach --target_k /
      --target_acc) if requested
    - saves a bar-with-error-bars comparison plot across seeds

Usage:
    python aggregate_seed_comparison.py \\
        --baseline-logs logs/A_347_200_s45_baseline.log logs/A_347_200_s46_baseline.log logs/A_347_200_s47_baseline.log \\
        --treatment-logs logs/A_347_200_s45_treatment.log logs/A_347_200_s46_treatment.log logs/A_347_200_s47_treatment.log \\
        --target_k 8 --target_acc 0.65

Logs must be paired by POSITION -- baseline-logs[i] and treatment-logs[i]
must be from the same seed.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from compare_baseline_vs_contrastive import parse_log, compute_convergence


def aggregate(logs, target_k=None, target_acc=None):
    """Parses a list of logs (one per seed) and collects final metrics +
    convergence epochs into arrays."""
    nmi, ari, acc, k = [], [], [], []
    k_epochs, acc_epochs = [], []

    for log_path in logs:
        run = parse_log(log_path)
        if run["final"] is None:
            print(f"WARNING: no final summary line found in {log_path} -- skipping this seed's final metrics.")
        else:
            nmi.append(run["final"]["nmi"])
            ari.append(run["final"]["ari"])
            acc.append(run["final"]["acc"])
            k.append(run["final"]["final_k"])

        conv = compute_convergence(run, target_k=target_k, target_acc=target_acc)
        k_epochs.append(conv["k_epoch"])       # may include None
        acc_epochs.append(conv["acc_epoch"])   # may include None

    return {
        "nmi": np.array(nmi), "ari": np.array(ari), "acc": np.array(acc), "k": np.array(k),
        "k_epochs": k_epochs, "acc_epochs": acc_epochs,
    }


def mean_std_str(arr):
    if len(arr) == 0:
        return "n/a"
    if len(arr) == 1:
        return f"{arr[0]:.4f} (n=1)"
    return f"{arr.mean():.4f} \u00b1 {arr.std(ddof=1):.4f}"


def convergence_summary_str(epochs_list, label):
    reached = [e for e in epochs_list if e is not None]
    n_not_reached = len(epochs_list) - len(reached)
    if len(reached) == 0:
        return f"{label}: never reached in {len(epochs_list)}/{len(epochs_list)} seeds"
    mean_e = np.mean(reached)
    std_e = np.std(reached, ddof=1) if len(reached) > 1 else 0.0
    note = f", not reached in {n_not_reached}/{len(epochs_list)} seed(s)" if n_not_reached else ""
    return f"{label}: epoch {mean_e:.1f} \u00b1 {std_e:.1f} (n={len(reached)}{note})"


def print_summary(baseline, treatment, n_seeds, target_k=None, target_acc=None):
    print("\n" + "=" * 72)
    print(f"AGGREGATE SUMMARY ACROSS {n_seeds} SEEDS")
    print("=" * 72)

    header = f"{'Metric':<10}{'Baseline':>26}{'Treatment':>26}"
    print(header)
    print("-" * len(header))
    for name, key in [("NMI", "nmi"), ("ARI", "ari"), ("ACC", "acc"), ("Final K", "k")]:
        b_str = mean_std_str(baseline[key])
        t_str = mean_std_str(treatment[key])
        print(f"{name:<10}{b_str:>26}{t_str:>26}")

    print("-" * len(header))
    print("PAIRED DIFFERENCE (treatment - baseline, per seed)")
    for name, key in [("NMI", "nmi"), ("ARI", "ari"), ("ACC", "acc"), ("Final K", "k")]:
        b_arr, t_arr = baseline[key], treatment[key]
        n = min(len(b_arr), len(t_arr))
        if n == 0:
            print(f"  {name}: n/a (missing final metrics)")
            continue
        diffs = t_arr[:n] - b_arr[:n]
        wins = int((diffs > 0).sum())
        diff_str = mean_std_str(diffs) if n > 1 else f"{diffs[0]:+.4f} (n=1)"
        print(f"  {name}: {diff_str}   |   treatment beat baseline on {wins}/{n} seed(s)")

    if target_k is not None or target_acc is not None:
        print("-" * len(header))
        print("CONVERGENCE SPEED")
        if target_k is not None:
            print("  " + convergence_summary_str(baseline["k_epochs"], f"Baseline K>={target_k}"))
            print("  " + convergence_summary_str(treatment["k_epochs"], f"Treatment K>={target_k}"))
        if target_acc is not None:
            print("  " + convergence_summary_str(baseline["acc_epochs"], f"Baseline ACC>={target_acc}"))
            print("  " + convergence_summary_str(treatment["acc_epochs"], f"Treatment ACC>={target_acc}"))

    print("=" * 72)
    print("Reading this: consistent direction across all seeds (win count == n_seeds "
          "or 0/n_seeds) with a paired-difference mean clearly larger than its std "
          "is good evidence of a real effect. A mixed win count (e.g. 2/3) or a "
          "paired-difference std comparable to or larger than its mean means the "
          "effect is not yet distinguishable from seed-to-seed noise at this sample size.")


def plot_aggregate(baseline, treatment, save_path="aggregate_comparison.png"):
    metrics = [("NMI", "nmi"), ("ARI", "ari"), ("ACC", "acc")]
    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(4.2 * (len(metrics) + 1), 4.5))

    for ax, (label, key) in zip(axes[:len(metrics)], metrics):
        b_arr, t_arr = baseline[key], treatment[key]
        b_mean, b_std = (b_arr.mean(), b_arr.std(ddof=1) if len(b_arr) > 1 else 0.0) if len(b_arr) else (0, 0)
        t_mean, t_std = (t_arr.mean(), t_arr.std(ddof=1) if len(t_arr) > 1 else 0.0) if len(t_arr) else (0, 0)
        ax.bar(["Baseline", "Treatment"], [b_mean, t_mean], yerr=[b_std, t_std],
               color=["tab:gray", "tab:blue"], capsize=6)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis="y")

    ax = axes[-1]
    b_arr, t_arr = baseline["k"], treatment["k"]
    b_mean, b_std = (b_arr.mean(), b_arr.std(ddof=1) if len(b_arr) > 1 else 0.0) if len(b_arr) else (0, 0)
    t_mean, t_std = (t_arr.mean(), t_arr.std(ddof=1) if len(t_arr) > 1 else 0.0) if len(t_arr) else (0, 0)
    ax.bar(["Baseline", "Treatment"], [b_mean, t_mean], yerr=[b_std, t_std],
           color=["tab:gray", "tab:blue"], capsize=6)
    ax.set_title("Final K")
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Baseline vs. Treatment, aggregated across seeds (mean \u00b1 std)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAggregate comparison plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline vs. treatment comparison across multiple seeds.")
    parser.add_argument("--baseline-logs", type=str, nargs="+", required=True,
                         help="Baseline log files, one per seed, same order as --treatment-logs.")
    parser.add_argument("--treatment-logs", type=str, nargs="+", required=True,
                         help="Treatment log files, one per seed, same order as --baseline-logs.")
    parser.add_argument("--target_k", type=int, default=None)
    parser.add_argument("--target_acc", type=float, default=None)
    parser.add_argument("--save-path", type=str, default="aggregate_comparison.png")
    args = parser.parse_args()

    if len(args.baseline_logs) != len(args.treatment_logs):
        raise ValueError(
            f"Got {len(args.baseline_logs)} baseline logs but {len(args.treatment_logs)} "
            f"treatment logs -- these must be paired 1:1 by seed."
        )

    baseline = aggregate(args.baseline_logs, target_k=args.target_k, target_acc=args.target_acc)
    treatment = aggregate(args.treatment_logs, target_k=args.target_k, target_acc=args.target_acc)

    print_summary(baseline, treatment, n_seeds=len(args.baseline_logs),
                  target_k=args.target_k, target_acc=args.target_acc)
    plot_aggregate(baseline, treatment, save_path=args.save_path)


if __name__ == "__main__":
    main()