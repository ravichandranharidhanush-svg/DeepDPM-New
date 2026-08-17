"""
visualize_rot_pair_data.py

UMAP separability check for datasets produced by generate_rot_pair_data.py.
Reads train_data.pt / train_labels.pt / metadata.json from an output
directory and produces a 3-panel plot (all classes / by digit / by
rotation) plus silhouette scores.

Fixed relative to the original snippet:
  - metadata.json is {"class_names": [...], "digits": [...],
    "rotations": [...], ...} -- class_names is a LIST indexed by class id,
    not a dict mapping label->idx. idx_to_label is now built with
    enumerate(class_names).
  - the variable previously named `rot_labels` was actually loading
    train_pair_labels.pt (the same/different pair label z, 1.0/0.0) --
    not rotation. Renamed to `pair_labels` for clarity; it's reported but
    not required for the plot (rotation is already recovered from the
    class name, same as before).
  - class/digit/rotation counts and labels are pulled from metadata
    instead of being hardcoded to 12 / 5 / 4 / "20 classes".

Usage:
    python visualize_rot_pair_data.py --out /path/to/ROT_PAIR_347_200_sam
"""

import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score


def Visualize(out, digits=None, rotations=None, reducer="pca"):
    data = torch.load(out + "/train_data.pt")          # (N, 2, D)
    labels = torch.load(out + "/train_labels.pt")       # (N,) class id
    pair_labels = torch.load(out + "/train_pair_labels.pt")  # (N,) 1.0/0.0 same-class

    with open(out + "/metadata.json") as f:
        meta = json.load(f)

    class_names = meta["class_names"]                  # list, index = class id
    idx_to_label = {i: name for i, name in enumerate(class_names)}
    n_classes = meta.get("n_classes", len(class_names))

    if digits is None:
        digits = meta.get("digits")
    if rotations is None:
        rotations = meta.get("rotations")
    n_digits = len(digits)
    n_rots = len(rotations)

    # Use only the FIRST embedding of each pair (the anchor)
    embs = data[:, 0, :].numpy()   # (N, D)
    lbl_np = labels.numpy()

    print(f"Loaded {len(embs)} points, {n_classes} classes "
          f"({n_digits} digits x {n_rots} rotations). "
          f"Pair labels: {int(pair_labels.sum())} same / "
          f"{int((1 - pair_labels).sum())} different out of {len(pair_labels)} pairs.")

    # ── Reduce to 2D just for plotting ──────────────────────────────────────
    # NOTE: if `embs` was already produced by UMAP (e.g. from
    # generate_embeddings_from_mnist_umap), running a SECOND UMAP here to
    # get down to 2D compounds two nonlinear reductions -- UMAP distances
    # aren't Euclidean-meaningful the way PCA's are, so re-embedding an
    # already-UMAP-reduced space with another UMAP is a common way to wash
    # out real structure, especially at smaller sample counts. Default to
    # PCA (linear -- can't introduce a second layer of manifold distortion)
    # so this plot reflects whether the SAVED embeddings are actually
    # separable, not an artifact of the visualization step. Pass
    # reducer="umap" to force UMAP instead (useful for direct raw-pixel or
    # synthetic-Gaussian embeddings, which haven't already been through UMAP).
    if embs.shape[1] == 2:
        embs_2d = embs
        print("embed_dim is already 2 -- plotting directly, no further reduction.")
    elif reducer == "umap":
        import umap
        reducer2d = umap.UMAP(n_components=2, random_state=42)
        embs_2d = reducer2d.fit_transform(embs)
    else:
        from sklearn.decomposition import PCA
        reducer2d = PCA(n_components=2, random_state=42)
        embs_2d = reducer2d.fit_transform(embs)
        print(f"PCA explained variance (top 2 PCs): {reducer2d.explained_variance_ratio_.sum():.3f}")

    # ── Plot 1 : All classes ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors_all = cm.get_cmap("tab20")(np.linspace(0, 1, max(n_classes, 2)))

    ax = axes[0]
    for cls_idx in range(n_classes):
        mask = lbl_np == cls_idx
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                   color=colors_all[cls_idx], label=idx_to_label[cls_idx],
                   s=25, alpha=0.7)
    ax.set_title(f"All {n_classes} Classes (digit + rotation)", fontsize=13)
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # ── Plot 2 : Colour by DIGIT only (ignore rotation) ────────────────────
    # class name format is "<digit>_<rotation_index>" (see generate_rot_pair_data.py)
    digit_labels_np = np.array([int(idx_to_label[i].split("_")[0]) for i in lbl_np])
    colors_digit = cm.get_cmap("tab10")(np.linspace(0, 0.5, max(n_digits, 2)))

    ax = axes[1]
    for i, digit in enumerate(digits):
        mask = digit_labels_np == digit
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                   color=colors_digit[i], label=f"digit {digit}", s=25, alpha=0.7)
    ax.set_title("Coloured by Digit (rotation ignored)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    # ── Plot 3 : Colour by ROTATION only (ignore digit) ────────────────────
    rot_idx_np = np.array([int(idx_to_label[i].split("_")[1]) for i in lbl_np])
    rot_names = [f"{r}\u00b0" for r in rotations]
    colors_rot = cm.get_cmap("Set1")(np.linspace(0, 0.4, max(n_rots, 2)))

    ax = axes[2]
    for r in range(n_rots):
        mask = rot_idx_np == r
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                   color=colors_rot[r], label=rot_names[r], s=25, alpha=0.7)
    ax.set_title("Coloured by Rotation (digit ignored)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.suptitle("UMAP Embedding Separability Check", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out + "/separability_plot.png", dpi=150)
    plt.show()

    # ── Quick silhouette score (quantitative separability) ──────────────────
    s_class = silhouette_score(embs, lbl_np, metric="euclidean")
    s_digit = silhouette_score(embs, digit_labels_np, metric="euclidean")
    s_rot = silhouette_score(embs, rot_idx_np, metric="euclidean")

    print("\nSilhouette Scores (higher = better separated, max=1.0):")
    print(f"  All {n_classes} classes : {s_class:.3f}")
    print(f"  By digit only  : {s_digit:.3f}")
    print(f"  By rotation    : {s_rot:.3f}")

    return {
        "silhouette_all_classes": s_class,
        "silhouette_digit": s_digit,
        "silhouette_rotation": s_rot,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a ROT_PAIR dataset's embedding separability.")
    parser.add_argument("--out", type=str, required=True, help="Directory containing train_data.pt / train_labels.pt / metadata.json")
    parser.add_argument("--reducer", type=str, default="pca", choices=["pca", "umap"],
                         help="2D reduction method for plotting. 'pca' (default) avoids compounding a second "
                              "nonlinear reduction on top of embeddings that already went through UMAP. Use "
                              "'umap' for embeddings that haven't been through UMAP yet (e.g. synthetic).")
    args = parser.parse_args()
    Visualize(args.out, reducer=args.reducer)