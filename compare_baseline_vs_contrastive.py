"""
compare_baseline_vs_contrastive.py

Runs two DeepDPM.py trainings back-to-back on the SAME dataset and seed --
one baseline (--contrastive_weight 0) and one treatment (your chosen
--contrastive_weight) -- captures their stdout, parses out the per-epoch
and final metrics DeepDPM.py already prints, and produces:

    - a comparison plot: K over epochs, NMI over epochs (baseline vs treatment)
    - a final-metrics bar chart (NMI / ARI / ACC / final K side by side)
    - a printed summary table

This relies entirely on the console output format already present in your
ClusterNetModel / DeepDPM.py (the periodic
"Epoch X | NMI: .. ARI: .. ACC: .. current K: N" lines and the final
"NMI: .., ARI: .., acc: .., final K: N" line) -- nothing extra needs to be
added to those files for this to work.

Usage:
    python compare_baseline_vs_contrastive.py \\
        --dir /content/DeepDPM_Final/Generated/Datasets/ROT_PAIR_347_200_sam \\
        --dataset custom_pair \\
        --max_epochs 300 --seed 45 --gpus 0 \\
        --contrastive_weight 1.0 \\
        --exp_name_prefix A_347_200

If you already have log files from separate runs (e.g. saved terminal
output), skip running and just compare existing logs:
    python compare_baseline_vs_contrastive.py --compare-only \\
        --baseline-log logs/baseline.log --treatment-log logs/treatment.log
"""

import argparse
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt


EPOCH_LINE_RE = re.compile(
    r"Epoch (\d+) \| NMI: ([\d.]+), ARI: ([\d.]+), ACC: ([\d.]+), current K: (\d+)"
)
FINAL_LINE_RE = re.compile(
    r"NMI:\s*([\d.]+),\s*ARI:\s*([\d.]+),\s*acc:\s*([\d.]+),\s*final K:\s*(\d+)"
)


# ---------------------------------------------------------------------------
# Run + capture
# ---------------------------------------------------------------------------
def run_and_log(cmd, log_path):
    """Runs `cmd`, streaming stdout to console AND saving it to log_path."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    print(f"\n{'=' * 70}\nRunning: {' '.join(cmd)}\nLogging to: {log_path}\n{'=' * 70}\n")

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()

    if process.returncode != 0:
        print(f"WARNING: process exited with code {process.returncode} -- check {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log(log_path):
    """Extracts per-epoch metrics and the final summary line from a log file."""
    epochs, nmis, aris, accs, ks = [], [], [], [], []
    final = None

    with open(log_path) as f:
        for line in f:
            m = EPOCH_LINE_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                nmis.append(float(m.group(2)))
                aris.append(float(m.group(3)))
                accs.append(float(m.group(4)))
                ks.append(int(m.group(5)))
                continue
            m = FINAL_LINE_RE.search(line)
            if m:
                final = {
                    "nmi": float(m.group(1)),
                    "ari": float(m.group(2)),
                    "acc": float(m.group(3)),
                    "final_k": int(m.group(4)),
                }

    return {
        "epochs": epochs, "nmi": nmis, "ari": aris, "acc": accs, "k": ks,
        "final": final,
    }


def compute_convergence(run, target_k=None, target_acc=None):
    """Finds the first epoch at which K and/or ACC reach the given targets.

    Returns a dict with 'k_epoch' and 'acc_epoch' (each an int epoch number
    or None if the target was never reached in the logged epochs). This is
    the actual "how fast did it get there" measure -- final-epoch numbers
    alone don't tell you whether a run reached a good clustering at epoch
    80 and held it, or only scraped past the target on the very last epoch.
    """
    result = {"k_epoch": None, "acc_epoch": None}

    if target_k is not None:
        for epoch, k in zip(run["epochs"], run["k"]):
            if k >= target_k:
                result["k_epoch"] = epoch
                break

    if target_acc is not None:
        for epoch, acc in zip(run["epochs"], run["acc"]):
            if acc >= target_acc:
                result["acc_epoch"] = epoch
                break

    return result


# ---------------------------------------------------------------------------
# Plotting + summary
# ---------------------------------------------------------------------------
def plot_comparison(baseline, treatment, save_path="logs/comparison_plot.png",
                     baseline_label="Baseline (contrastive_weight=0)",
                     treatment_label="Treatment",
                     baseline_conv=None, treatment_conv=None, target_k=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.step(baseline["epochs"], baseline["k"], where="post", label=baseline_label, color="tab:gray")
    ax.step(treatment["epochs"], treatment["k"], where="post", label=treatment_label, color="tab:blue")
    if target_k is not None:
        ax.axhline(target_k, color="black", linestyle=":", linewidth=1, alpha=0.6, label=f"target K={target_k}")
    if baseline_conv and baseline_conv.get("k_epoch") is not None:
        ax.axvline(baseline_conv["k_epoch"], color="tab:gray", linestyle="--", linewidth=1, alpha=0.7)
    if treatment_conv and treatment_conv.get("k_epoch") is not None:
        ax.axvline(treatment_conv["k_epoch"], color="tab:blue", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("K (number of clusters)")
    ax.set_title("Cluster count over training")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(baseline["epochs"], baseline["nmi"], label=baseline_label, color="tab:gray", marker="o", markersize=3)
    ax.plot(treatment["epochs"], treatment["nmi"], label=treatment_label, color="tab:blue", marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("NMI")
    ax.set_title("NMI over training")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("Baseline vs. Treatment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved to: {save_path}")


def plot_final_bars(baseline, treatment, save_path="logs/comparison_final_bars.png",
                     baseline_label="Baseline", treatment_label="Treatment"):
    if baseline["final"] is None or treatment["final"] is None:
        print("Skipping final-metrics bar chart -- final summary line not found in one or both logs.")
        return

    metrics = ["nmi", "ari", "acc"]
    b_vals = [baseline["final"][m] for m in metrics]
    t_vals = [treatment["final"][m] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], b_vals, width, label=baseline_label, color="tab:gray")
    ax.bar([i + width / 2 for i in x], t_vals, width, label=treatment_label, color="tab:blue")
    ax.set_xticks(list(x)); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Final NMI / ARI / ACC")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(["Baseline", "Treatment"], [baseline["final"]["final_k"], treatment["final"]["final_k"]],
           color=["tab:gray", "tab:blue"])
    ax.set_title("Final K")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Final-metrics bar chart saved to: {save_path}")


def print_summary(baseline, treatment, baseline_label="Baseline", treatment_label="Treatment",
                   baseline_conv=None, treatment_conv=None, target_k=None, target_acc=None):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Metric':<10}{baseline_label:>20}{treatment_label:>20}"
    print(header)
    print("-" * len(header))
    if baseline["final"] and treatment["final"]:
        for m, label in [("nmi", "NMI"), ("ari", "ARI"), ("acc", "ACC"), ("final_k", "Final K")]:
            b = baseline["final"][m]
            t = treatment["final"][m]
            print(f"{label:<10}{b:>20}{t:>20}")
    else:
        print("Final summary line missing from one or both logs -- "
              "check that training completed and printed the final "
              "'NMI: .., ARI: .., acc: .., final K: ..' line.")

    if (target_k is not None or target_acc is not None) and baseline_conv and treatment_conv:
        print("-" * len(header))
        print("CONVERGENCE SPEED (first epoch reaching target)")
        if target_k is not None:
            b_e = baseline_conv["k_epoch"]
            t_e = treatment_conv["k_epoch"]
            b_str = f"epoch {b_e}" if b_e is not None else "not reached"
            t_str = f"epoch {t_e}" if t_e is not None else "not reached"
            print(f"{'K>=' + str(target_k):<10}{b_str:>20}{t_str:>20}")
        if target_acc is not None:
            b_e = baseline_conv["acc_epoch"]
            t_e = treatment_conv["acc_epoch"]
            b_str = f"epoch {b_e}" if b_e is not None else "not reached"
            t_str = f"epoch {t_e}" if t_e is not None else "not reached"
            print(f"{'ACC>=' + str(target_acc):<10}{b_str:>20}{t_str:>20}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs. contrastive-enabled DeepDPM training.")
    parser.add_argument("--compare-only", action="store_true",
                         help="Skip running training; just parse+compare existing log files.")
    parser.add_argument("--baseline-log", type=str, default=None,
                         help="Path to save/read the baseline log. If not set and --compare-only is NOT used, "
                              "defaults to logs/<exp_name_prefix>_baseline.log so different runs (e.g. different "
                              "seeds) don't silently overwrite each other's logs. Required if --compare-only is set.")
    parser.add_argument("--treatment-log", type=str, default=None,
                         help="Path to save/read the treatment log. Same auto-derivation as --baseline-log.")

    # training args (only used unless --compare-only)
    parser.add_argument("--dir", type=str, help="Dataset directory (--dir passed to DeepDPM.py)")
    parser.add_argument("--dataset", type=str, default="custom_pair")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--contrastive_weight", type=float, default=1.0,
                         help="Contrastive weight to use for the TREATMENT run. Baseline always uses 0.")
    parser.add_argument("--exp_name_prefix", type=str, default="compare")
    parser.add_argument("--deepdpm_script", type=str, default="DeepDPM.py")
    parser.add_argument("--extra_args", type=str, default="--use_labels_for_eval --offline",
                         help="Extra args appended verbatim to both commands.")
    parser.add_argument("--target_k", type=int, default=None,
                         help="If set, reports the first epoch each run's K reaches this value (convergence speed).")
    parser.add_argument("--target_acc", type=float, default=None,
                         help="If set, reports the first epoch each run's ACC reaches this value (convergence speed).")

    args = parser.parse_args()

    if args.compare_only:
        if args.baseline_log is None or args.treatment_log is None:
            print("ERROR: --compare-only requires explicit --baseline-log and --treatment-log paths "
                  "(there's no training run here to derive a default from).")
            sys.exit(1)
    else:
        # Auto-derive from exp_name_prefix if not explicitly given -- this is
        # what prevents different runs (e.g. different seeds) from silently
        # overwriting each other's logs at a shared default path.
        if args.baseline_log is None:
            args.baseline_log = f"logs/{args.exp_name_prefix}_baseline.log"
        if args.treatment_log is None:
            args.treatment_log = f"logs/{args.exp_name_prefix}_treatment.log"

        if os.path.exists(args.baseline_log) or os.path.exists(args.treatment_log):
            print(f"WARNING: {args.baseline_log} or {args.treatment_log} already exists and will be "
                  f"OVERWRITTEN. If this is a different run (e.g. a new seed), use a different "
                  f"--exp_name_prefix or pass --baseline-log/--treatment-log explicitly.")

    if not args.compare_only:
        if not args.dir:
            print("ERROR: --dir is required unless --compare-only is set.")
            sys.exit(1)

        common = [
            sys.executable, args.deepdpm_script,
            "--dir", args.dir,
            "--dataset", args.dataset,
            "--max_epochs", str(args.max_epochs),
            "--seed", str(args.seed),
            "--gpus", args.gpus,
        ] + args.extra_args.split()

        baseline_cmd = common + [
            "--exp_name", f"{args.exp_name_prefix}_baseline",
            "--contrastive_weight", "0",
        ]
        treatment_cmd = common + [
            "--exp_name", f"{args.exp_name_prefix}_treatment",
            "--contrastive_weight", str(args.contrastive_weight),
        ]

        run_and_log(baseline_cmd, args.baseline_log)
        run_and_log(treatment_cmd, args.treatment_log)

    baseline = parse_log(args.baseline_log)
    treatment = parse_log(args.treatment_log)

    baseline_label = "Baseline (contrastive_weight=0)"
    treatment_label = f"Treatment (contrastive_weight={args.contrastive_weight})" if not args.compare_only else "Treatment"

    baseline_conv = compute_convergence(baseline, target_k=args.target_k, target_acc=args.target_acc)
    treatment_conv = compute_convergence(treatment, target_k=args.target_k, target_acc=args.target_acc)

    print_summary(baseline, treatment, baseline_label, treatment_label,
                   baseline_conv=baseline_conv, treatment_conv=treatment_conv,
                   target_k=args.target_k, target_acc=args.target_acc)
    plot_comparison(baseline, treatment, baseline_label=baseline_label, treatment_label=treatment_label,
                     baseline_conv=baseline_conv, treatment_conv=treatment_conv, target_k=args.target_k)
    plot_final_bars(baseline, treatment, baseline_label=baseline_label, treatment_label=treatment_label)


if __name__ == "__main__":
    main()