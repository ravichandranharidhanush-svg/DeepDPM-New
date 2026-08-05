"""
generate_rot_pair_data.py

Synthetic data generator reproducing the ROT_PAIR_<digits>_<samples_per_class>
structure used throughout this debugging session, saved in the exact .pt
layout `CustomPairDataset` (in src/datasets.py) expects:

    <out_dir>/train_data.pt          FloatTensor (N, 2, D)  -- [anchor, partner]
    <out_dir>/train_labels.pt        LongTensor  (N,)       -- anchor's class id
    <out_dir>/train_pair_labels.pt   FloatTensor (N,)        -- 1.0 same class, 0.0 different
    <out_dir>/test_data.pt           (optional, same layout)
    <out_dir>/test_labels.pt
    <out_dir>/test_pair_labels.pt
    <out_dir>/metadata.json          class_names + generation params (for your reference only;
                                      not read by CustomPairDataset)

Structure of the embeddings: digit identity and rotation are two
near-orthogonal axes of variation, matching the clean 12-cluster
separability seen in the UMAP check. 3 digits x 4 rotations = 12 classes,
`--samples-per-class` points per class.

NOTE: this is a reconstruction, not your original generator (that file was
never shared in this conversation) -- but it now matches the exact on-disk
convention `CustomPairDataset` reads, so it should be a drop-in dataset for
testing the full pipeline end-to-end.

Usage:
    python generate_rot_pair_data.py \\
        --digits 3 4 7 \\
        --rotations 0 90 180 270 \\
        --samples-per-class 200 \\
        --embed-dim 10 \\
        --seed 45 \\
        --out-dir /content/DeepDPM_Final/Generated/Datasets/ROT_PAIR_347_200_sam

Then run:
    python DeepDPM.py --dir <out-dir> --dataset custom_pair \\
        --max_epochs 300 --seed 45 --gpus 0 --exp_name A_347_200 \\
        --use_labels_for_eval --contrastive_weight 1.0
"""

import argparse
import json
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Embedding generation (unpaired base codes)
# ---------------------------------------------------------------------------
def generate_embeddings(
    digits,
    rotations,
    samples_per_class=200,
    embed_dim=10,
    digit_sep=8.0,
    rot_sep=4.0,
    noise_std=1.0,
    seed=45,
):
    """Generate synthetic (digit, rotation) embeddings.

    Each digit gets its own random direction in embedding space; each
    rotation gets its own random direction, orthogonalized against the
    digit directions. A class's cluster center is
    (digit_center + rotation_center); points are drawn from an isotropic
    Gaussian around that center.

    Returns:
        codes: (N, embed_dim) float32 array
        labels: (N,) int64 array, combined class index in
                [0, len(digits)*len(rotations))
        class_names: list of "digit_rotationIndex" strings, ordered so
                     class_names[labels[i]] describes point i
    """
    rng = np.random.default_rng(seed)

    n_digits = len(digits)
    n_rots = len(rotations)

    if embed_dim < max(n_digits, n_rots):
        raise ValueError(
            f"embed_dim ({embed_dim}) must be >= max(len(digits), len(rotations)) "
            f"({max(n_digits, n_rots)}) for the directions to stay separable."
        )

    raw = rng.normal(size=(n_digits + n_rots, embed_dim))
    q, _ = np.linalg.qr(raw.T)
    directions = q.T[: n_digits + n_rots]
    digit_dirs = directions[:n_digits]
    rot_dirs = directions[n_digits:n_digits + n_rots]

    codes = []
    labels = []
    class_names = []

    class_idx = 0
    for di, digit in enumerate(digits):
        for ri, rot in enumerate(rotations):
            center = digit_sep * digit_dirs[di] + rot_sep * rot_dirs[ri]
            pts = center[None, :] + noise_std * rng.normal(size=(samples_per_class, embed_dim))
            codes.append(pts)
            labels.append(np.full(samples_per_class, class_idx, dtype=np.int64))
            class_names.append(f"{digit}_{ri}")
            class_idx += 1

    codes = np.concatenate(codes, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0)

    # digit/rotation index per point, derived the same way class_names was
    # built above (class_idx = di * n_rots + ri) -- needed for same-digit
    # pairing further down.
    digit_idx = labels // n_rots
    rot_idx = labels % n_rots

    perm = rng.permutation(len(codes))
    codes, labels = codes[perm], labels[perm]
    digit_idx, rot_idx = digit_idx[perm], rot_idx[perm]

    return codes, labels, class_names, digit_idx, rot_idx


# ---------------------------------------------------------------------------
# Pair construction -- materializes actual (anchor, partner, z) triples
# ---------------------------------------------------------------------------
def build_pairs(codes, labels, pairs_per_sample=1, positive_prob=0.5, seed=45):
    """For each point in `codes`, build `pairs_per_sample` (anchor, partner)
    pairs, each independently drawn as positive (same class) with
    probability `positive_prob`, else negative (different class).

    Returns:
        paired_codes: (M, 2, D) float32 array, M = N * pairs_per_sample
        paired_labels: (M,) int64 array -- the anchor's class
        pair_labels: (M,) float32 array -- 1.0 if same class, 0.0 if different
    """
    rng = np.random.default_rng(seed)
    n = len(codes)

    by_class = {}
    for c in np.unique(labels):
        by_class[c] = np.where(labels == c)[0]
    all_classes = list(by_class.keys())

    paired_codes = []
    paired_labels = []
    pair_labels = []

    for _ in range(pairs_per_sample):
        for i in range(n):
            anchor_label = labels[i]
            is_positive = rng.random() < positive_prob

            if is_positive:
                candidates = by_class[anchor_label]
                # avoid trivially pairing a point with itself when possible
                if len(candidates) > 1:
                    partner_idx = i
                    while partner_idx == i:
                        partner_idx = int(rng.choice(candidates))
                else:
                    partner_idx = i
                z = 1.0
            else:
                other_classes = [c for c in all_classes if c != anchor_label]
                partner_class = rng.choice(other_classes)
                partner_idx = int(rng.choice(by_class[partner_class]))
                z = 0.0

            paired_codes.append(np.stack([codes[i], codes[partner_idx]], axis=0))
            paired_labels.append(anchor_label)
            pair_labels.append(z)

    paired_codes = np.stack(paired_codes, axis=0).astype(np.float32)   # (M, 2, D)
    paired_labels = np.array(paired_labels, dtype=np.int64)            # (M,)
    pair_labels = np.array(pair_labels, dtype=np.float32)              # (M,)

    # shuffle across the whole set so repeated-anchor pairs aren't adjacent
    perm = rng.permutation(len(paired_codes))
    return paired_codes[perm], paired_labels[perm], pair_labels[perm]


def build_pairs_same_digit(codes, digit_idx, rot_idx, labels, pairs_per_sample=1, positive_prob=0.5, seed=45):
    """Same-digit-restricted pairing.

    Both anchor and partner are always drawn from the SAME digit. z=1 if
    they also share rotation (positive: same full class), z=0 if they
    differ in rotation (negative: same digit, different rotation).

    This is deliberately different from build_pairs(): by construction it
    can never encode "different digit" (the distinction the unsupervised
    GMM/split-merge machinery already finds trivially on its own -- see
    the digit-only baseline's K=11, ACC=0.82 result). Every pair here
    carries information specifically about the harder, finer-grained
    rotation distinction, which is the complementary signal worth adding.

    Intended to be paired with --contrastive_weight 0
    --subcluster_contrastive_weight X in ClusterNetModel/DeepDPM.py, so
    the pairwise supervision only influences the subclustering net (which
    is what's responsible for finding structure *within* an already
    top-level-clustered digit) rather than the top-level cluster_net.

    Returns:
        paired_codes: (M, 2, D) float32
        paired_labels: (M,) int64 -- anchor's full class id
        pair_labels: (M,) float32 -- 1.0 same rotation, 0.0 different rotation
    """
    rng = np.random.default_rng(seed)
    n = len(codes)

    by_digit = {}
    for d in np.unique(digit_idx):
        by_digit[d] = np.where(digit_idx == d)[0]

    paired_codes = []
    paired_labels = []
    pair_labels = []

    for _ in range(pairs_per_sample):
        for i in range(n):
            d = digit_idx[i]
            r = rot_idx[i]
            same_digit_idxs = by_digit[d]

            is_positive = rng.random() < positive_prob

            if is_positive:
                candidates = same_digit_idxs[rot_idx[same_digit_idxs] == r]
                if len(candidates) > 1:
                    partner_idx = i
                    while partner_idx == i:
                        partner_idx = int(rng.choice(candidates))
                else:
                    partner_idx = i
                z = 1.0
            else:
                candidates = same_digit_idxs[rot_idx[same_digit_idxs] != r]
                if len(candidates) == 0:
                    # degenerate (shouldn't happen with >1 rotation), fall back to positive
                    partner_idx = i
                    z = 1.0
                else:
                    partner_idx = int(rng.choice(candidates))
                    z = 0.0

            paired_codes.append(np.stack([codes[i], codes[partner_idx]], axis=0))
            paired_labels.append(labels[i])
            pair_labels.append(z)

    paired_codes = np.stack(paired_codes, axis=0).astype(np.float32)
    paired_labels = np.array(paired_labels, dtype=np.int64)
    pair_labels = np.array(pair_labels, dtype=np.float32)

    perm = rng.permutation(len(paired_codes))
    return paired_codes[perm], paired_labels[perm], pair_labels[perm]


# ---------------------------------------------------------------------------
# Save (.pt layout expected by CustomPairDataset) / metadata
# ---------------------------------------------------------------------------
def save_pt_dataset(out_dir, paired_codes, paired_labels, pair_labels, split="train"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.from_numpy(paired_codes), os.path.join(out_dir, f"{split}_data.pt"))
    torch.save(torch.from_numpy(paired_labels), os.path.join(out_dir, f"{split}_labels.pt"))
    torch.save(torch.from_numpy(pair_labels), os.path.join(out_dir, f"{split}_pair_labels.pt"))
    print(f"Saved {split} split to {out_dir}:")
    print(f"  {split}_data.pt        {tuple(paired_codes.shape)}")
    print(f"  {split}_labels.pt      {tuple(paired_labels.shape)}")
    print(f"  {split}_pair_labels.pt {tuple(pair_labels.shape)}  "
          f"(positives: {int(pair_labels.sum())}/{len(pair_labels)})")


def save_metadata(out_dir, class_names, meta_extra=None):
    meta = {"n_classes": len(class_names), "class_names": class_names}
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ROT_PAIR digit+rotation embeddings (.pt layout).")
    parser.add_argument("--digits", type=int, nargs="+", default=[3, 4, 7])
    parser.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--digit-sep", type=float, default=8.0)
    parser.add_argument("--rot-sep", type=float, default=4.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--pairs-per-sample", type=int, default=1, help="How many (anchor, partner) pairs to generate per base point")
    parser.add_argument("--positive-prob", type=float, default=0.5, help="Probability a generated pair is 'positive' under the chosen --pairing-mode")
    parser.add_argument("--pairing-mode", type=str, default="random", choices=["random", "same-digit"],
                         help="'random': pairs from anywhere (z=1 same full class, z=0 different class -- redundant with what unsupervised clustering already finds easily). "
                              "'same-digit': pairs restricted to the same digit (z=1 same rotation, z=0 different rotation -- targets the harder, complementary distinction; intended for use with --subcluster_contrastive_weight rather than --contrastive_weight).")
    parser.add_argument("--test-fraction", type=float, default=0.0, help="Fraction of base points held out for a test split (0 = no test split, matching your original runs' 'Test data not found' behavior)")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    codes, labels, class_names, digit_idx, rot_idx = generate_embeddings(
        digits=args.digits,
        rotations=args.rotations,
        samples_per_class=args.samples_per_class,
        embed_dim=args.embed_dim,
        digit_sep=args.digit_sep,
        rot_sep=args.rot_sep,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    if args.test_fraction > 0:
        rng = np.random.default_rng(args.seed)
        n = len(codes)
        perm = rng.permutation(n)
        n_test = int(n * args.test_fraction)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        train_codes, train_labels = codes[train_idx], labels[train_idx]
        train_digit_idx, train_rot_idx = digit_idx[train_idx], rot_idx[train_idx]
        test_codes, test_labels = codes[test_idx], labels[test_idx]
        test_digit_idx, test_rot_idx = digit_idx[test_idx], rot_idx[test_idx]
    else:
        train_codes, train_labels = codes, labels
        train_digit_idx, train_rot_idx = digit_idx, rot_idx
        test_codes, test_labels = None, None
        test_digit_idx, test_rot_idx = None, None

    def _build(codes_, labels_, digit_idx_, rot_idx_, seed_):
        if args.pairing_mode == "same-digit":
            return build_pairs_same_digit(
                codes_, digit_idx_, rot_idx_, labels_,
                pairs_per_sample=args.pairs_per_sample,
                positive_prob=args.positive_prob,
                seed=seed_,
            )
        return build_pairs(
            codes_, labels_,
            pairs_per_sample=args.pairs_per_sample,
            positive_prob=args.positive_prob,
            seed=seed_,
        )

    train_paired_codes, train_paired_labels, train_pair_labels = _build(
        train_codes, train_labels, train_digit_idx, train_rot_idx, args.seed
    )
    save_pt_dataset(args.out_dir, train_paired_codes, train_paired_labels, train_pair_labels, split="train")

    if test_codes is not None and len(test_codes) > 0:
        test_paired_codes, test_paired_labels, test_pair_labels = _build(
            test_codes, test_labels, test_digit_idx, test_rot_idx, args.seed + 1
        )
        save_pt_dataset(args.out_dir, test_paired_codes, test_paired_labels, test_pair_labels, split="test")
    else:
        print("No test split generated (--test-fraction 0). "
              "CustomPairDataset will print 'Test data not found! running only with train data', matching your original runs.")

    save_metadata(
        args.out_dir, class_names,
        meta_extra={
            "digits": args.digits,
            "rotations": args.rotations,
            "samples_per_class": args.samples_per_class,
            "embed_dim": args.embed_dim,
            "digit_sep": args.digit_sep,
            "rot_sep": args.rot_sep,
            "noise_std": args.noise_std,
            "pairs_per_sample": args.pairs_per_sample,
            "positive_prob": args.positive_prob,
            "pairing_mode": args.pairing_mode,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()