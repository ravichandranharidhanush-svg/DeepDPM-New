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
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score


def Visualize(out, digits=None, rotations=None, reducer="pca"):
    data = torch.load(out + "/train_data.pt")
    labels = torch.load(out + "/train_labels.pt")

    # Auto-detect paired (N, 2, D) vs unpaired (N, D) layout -- the newer
    # generator (generate_mnist_rotation_pairs.py) saves BOTH: the paired
    # subset at <out>/train_data.pt, and the full, unpaired dataset at
    # <out>/unpaired/train_data.pt. The unpaired one has no
    # train_pair_labels.pt (there's no pairing to report) and usually has
    # far more points, since it isn't limited to whatever a restrictive
    # --rotation-pairs list happened to select as anchors/partners.
    if data.ndim == 3:
        embs = data[:, 0, :].numpy()   # paired: use only the anchor view
        pair_labels_path = out + "/train_pair_labels.pt"
        try:
            pair_labels = torch.load(pair_labels_path)
            pair_info = (f"Pair labels: {int(pair_labels.sum())} same / "
                         f"{int((1 - pair_labels).sum())} different out of {len(pair_labels)} pairs.")
        except FileNotFoundError:
            pair_info = "(no train_pair_labels.pt found)"
    else:
        embs = data.numpy()            # unpaired: already flat
        pair_info = "(unpaired dataset -- no pair labels)"

    lbl_np = labels.numpy()

    # metadata.json is only saved once, at the top-level out_dir -- if
    # visualizing the unpaired/ subfolder directly, fall back to the parent.
    meta_path = out + "/metadata.json"
    if not os.path.exists(meta_path):
        parent_meta_path = os.path.join(os.path.dirname(out.rstrip("/")), "metadata.json")
        if os.path.exists(parent_meta_path):
            meta_path = parent_meta_path
    with open(meta_path) as f:
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

    print(f"Loaded {len(embs)} points, {n_classes} classes "
          f"({n_digits} digits x {n_rots} rotations). {pair_info}")

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


def VisualizeByDigit(out, digit, digits=None, rotations=None, reducer="pca"):
    """Filters to a single digit's points and reduces ONLY those to 2D,
    colored by rotation. Digit-level variance dominates a global (all-digit)
    2D projection, which can visually hide real rotation sub-structure even
    when it's genuinely present in the full-dimensional embedding (a
    non-trivial rotation silhouette computed on the full data doesn't
    guarantee it's visible in a 2D view where digit separation eats most of
    the available projection "budget"). Removing digit variance from the
    picture lets rotation become the dominant remaining signal, so any real
    within-digit clustering should become visible here if it exists.
    """
    data = torch.load(out + "/train_data.pt")
    labels = torch.load(out + "/train_labels.pt")
    embs = (data[:, 0, :].numpy() if data.ndim == 3 else data.numpy())
    lbl_np = labels.numpy()

    meta_path = out + "/metadata.json"
    if not os.path.exists(meta_path):
        parent_meta_path = os.path.join(os.path.dirname(out.rstrip("/")), "metadata.json")
        if os.path.exists(parent_meta_path):
            meta_path = parent_meta_path
    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    if digits is None:
        digits = meta.get("digits")
    if rotations is None:
        rotations = meta.get("rotations")
    n_rots = len(rotations)

    if digit not in digits:
        raise ValueError(f"digit {digit} not in dataset's digits {digits}")
    digit_i = digits.index(digit)
    # class ids for this digit span [digit_i*n_rots, digit_i*n_rots + n_rots)
    mask = (lbl_np >= digit_i * n_rots) & (lbl_np < (digit_i + 1) * n_rots)
    sub_embs = embs[mask]
    sub_labels = lbl_np[mask]
    rot_idx_np = sub_labels - digit_i * n_rots   # 0..n_rots-1

    if len(sub_embs) < 5:
        print(f"Only {len(sub_embs)} points for digit {digit} -- too few to visualize meaningfully.")
        return None

    print(f"Digit {digit}: {len(sub_embs)} points, reducing ONLY this subset to 2D "
          f"(removes digit-level variance from the projection)...")

    if sub_embs.shape[1] == 2:
        embs_2d = sub_embs
    elif reducer == "umap":
        import umap
        embs_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(sub_embs)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embs_2d = pca.fit_transform(sub_embs)
        print(f"  PCA explained variance (top 2 PCs, digit {digit} only): {pca.explained_variance_ratio_.sum():.3f}")

    rot_names = [f"{r}\u00b0" for r in rotations]
    colors_rot = cm.get_cmap("Set1")(np.linspace(0, 0.4, max(n_rots, 2)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in range(n_rots):
        m = rot_idx_np == r
        ax.scatter(embs_2d[m, 0], embs_2d[m, 1], color=colors_rot[r], label=rot_names[r], s=30, alpha=0.7)
    ax.set_title(f"Digit {digit} only, coloured by rotation ({len(sub_embs)} points)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    plt.tight_layout()
    save_path = f"{out}/digit_{digit}_rotation_only.png"
    plt.savefig(save_path, dpi=150)
    plt.show()

    if n_rots > 1:
        s = silhouette_score(sub_embs, rot_idx_np, metric="euclidean")
        print(f"  Rotation silhouette WITHIN digit {digit} only: {s:.3f}")
        return s
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a ROT_PAIR dataset's embedding separability.")
    parser.add_argument("--out", type=str, required=True, help="Directory containing train_data.pt / train_labels.pt / metadata.json")
    parser.add_argument("--reducer", type=str, default="pca", choices=["pca", "umap"],
                         help="2D reduction method for plotting. 'pca' (default) avoids compounding a second "
                              "nonlinear reduction on top of embeddings that already went through UMAP. Use "
                              "'umap' for embeddings that haven't been through UMAP yet (e.g. synthetic).")
    args = parser.parse_args()
    Visualize(args.out, reducer=args.reducer)