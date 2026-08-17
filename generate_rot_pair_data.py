"""
generate_mnist_rotation_pairs.py

Clean, single-purpose data generator:
    1. Collects real MNIST digit images for the requested --digits
    2. Applies the rotations implied by --rotation-pairs (only the angles
       actually used -- nothing wasted generating unused rotations)
    3. Trains a small CNN to embed the rotated images -- a digit-identity
       head (main) + a rotation-identity head (auxiliary), so the
       resulting embedding is genuinely digit-separable while still
       retaining real rotation-relevant structure (a pure digit-only
       classifier becomes rotation-INVARIANT by design and erases
       rotation information entirely -- see rotation_aux_weight below)
    4. Builds pairs using EXACTLY the (rotation_a, rotation_b) combinations
       you specify in --rotation-pairs -- e.g. a cyclic pattern like
       "0-90,90-180,180-270,270-0" -- nothing else, per digit
    5. Saves BOTH the paired dataset (for --dataset custom_pair) AND the
       full unpaired dataset (for --dataset custom) -- the unpaired export
       matters because a restrictive --rotation-pairs list can select only
       a small fraction of the full embedding pool as anchors/partners;
       the unpaired export lets you sanity-check clustering on the FULL
       data, independent of that subsetting.
    6. Prints a sample of generated pairs so you can eyeball the pairing
       logic before training.

Output layout (<out_dir>):
    train_data.pt          FloatTensor (M, 2, D)  -- paired: [anchor, partner]
    train_labels.pt        LongTensor  (M,)       -- anchor's full class id
    train_pair_labels.pt   FloatTensor (M,)       -- 1.0 same rotation, 0.0 different
    unpaired/train_data.pt FloatTensor (N, D)     -- ALL embedded points, no pairing subsetting
    unpaired/train_labels.pt LongTensor (N,)
    metadata.json           class_names, digits, rotations, rotation_pairs, generation params

Usage:
    python generate_mnist_rotation_pairs.py \\
        --digits 3 4 7 \\
        --rotation-pairs "0-90,90-180,180-270,270-0" \\
        --samples-per-class 200 --embed-dim 10 \\
        --cnn-epochs 8 --rotation-aux-weight 0.3 \\
        --pairs-per-combo 5 --seed 45 \\
        --out-dir /content/DeepDPM-New/Generated/Datasets/ROT_PAIR_MNIST_347_cyclic

Then train with e.g.:
    python DeepDPM.py --dir <out_dir> --dataset custom_pair \\
        --max_epochs 300 --seed 45 --gpus 0 --use_labels_for_eval --offline \\
        --exp_name mnist_cyclic --contrastive_weight 1.0

Or sanity-check the full unpaired data first:
    python DeepDPM.py --dir <out_dir>/unpaired --dataset custom \\
        --max_epochs 300 --seed 45 --gpus 0 --use_labels_for_eval --offline \\
        --exp_name mnist_cyclic_unpaired_check --contrastive_weight 0

Requires: torch, torchvision, numpy. Downloading MNIST requires normal
internet access (works in Colab; will fail in network-sandboxed
environments that block torchvision's hosting mirrors).
"""

import argparse
import json
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Step 1-3: MNIST -> rotated images -> trained CNN embedding
# ---------------------------------------------------------------------------
def generate_mnist_rotation_embeddings(
    digits,
    rotations,
    samples_per_class=200,
    embed_dim=10,
    seed=45,
    mnist_root="./mnist_data",
    cnn_epochs=15,
    cnn_lr=1e-3,
    cnn_weight_decay=1e-4,
    cnn_batch_size=64,
    cnn_dropout=0.3,
    val_fraction=0.15,
    rotation_aux_weight=0.3,
    l2_normalize_embed=True,
    device=None,
):
    """Downloads MNIST, builds `samples_per_class` rotated images for every
    (digit, rotation) combination, trains a two-head CNN (digit + rotation),
    and returns the penultimate-layer embedding for every image.

    Architecture: 3 conv blocks (Conv2d+BatchNorm+ReLU+MaxPool, channels
    16->32->64) -> global average pool -> FC embedding layer -> dropout ->
    two linear heads (digit, rotation). BatchNorm stabilizes training,
    global average pooling (instead of flattening the full spatial map)
    makes the embedding less sensitive to exactly WHERE features land
    after rotation (encouraging genuinely rotation-relevant content rather
    than raw pixel position), and dropout regularizes against overfitting
    to the small architecture. If l2_normalize_embed=True (default), the
    final embedding is L2-normalized -- this is standard practice for
    clustering-quality embeddings: it keeps distances meaningful and
    prevents a few high-magnitude dimensions from dominating downstream
    Euclidean-distance-based clustering (which raw, unnormalized features
    are prone to).

    rotation_aux_weight: 0 = pure digit classifier -- becomes rotation-
    invariant by design, tends to erase rotation-discriminating structure
    from the embedding entirely (confirmed empirically: rotation silhouette
    near 0). >0 adds a jointly-trained auxiliary head that also predicts
    rotation, keeping the shared embedding predictive of both factors.
    0.3 is a reasonable starting point; raise it if rotation structure
    still isn't showing up in the resulting embedding, lower it if digit
    separation degrades too much.

    Returns:
        codes: (N, embed_dim) float32
        labels: (N,) int64 -- combined class id, class_idx = digit_i * n_rot + rot_i
        class_names: list of "digit_rotationIndex" strings
        digit_idx: (N,) int64 -- index into `digits`
        rot_idx: (N,) int64 -- index into `rotations`
    """
    try:
        import torchvision
        import torchvision.transforms.functional as TF
    except ImportError as e:
        raise ImportError("This script requires torchvision: pip install torchvision") from e
    import torch.nn as nn
    import torch.nn.functional as Fnn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading MNIST (root={mnist_root}, download if needed)...")
    dataset = torchvision.datasets.MNIST(root=mnist_root, train=True, download=True)

    n_digits = len(digits)
    n_rots = len(rotations)

    all_images = []
    labels = []
    digit_idx_list = []
    rot_idx_list = []
    class_names = []

    class_idx = 0
    for di, digit in enumerate(digits):
        pool_mask = dataset.targets == digit
        pool = dataset.data[pool_mask]
        if len(pool) < samples_per_class:
            raise ValueError(f"Not enough MNIST images for digit {digit}: requested "
                              f"{samples_per_class}, only {len(pool)} available.")
        chosen = rng.choice(len(pool), size=samples_per_class, replace=False)
        base_imgs = pool[chosen].float()

        for ri, rot in enumerate(rotations):
            rotated = torch.stack([
                TF.rotate(img.unsqueeze(0), angle=float(rot)).squeeze(0)
                for img in base_imgs
            ])
            all_images.append((rotated / 255.0).unsqueeze(1))
            labels.append(np.full(samples_per_class, class_idx, dtype=np.int64))
            digit_idx_list.append(np.full(samples_per_class, di, dtype=np.int64))
            rot_idx_list.append(np.full(samples_per_class, ri, dtype=np.int64))
            class_names.append(f"{digit}_{ri}")
            class_idx += 1

    all_images = torch.cat(all_images, dim=0)
    labels = np.concatenate(labels)
    digit_idx = np.concatenate(digit_idx_list)
    rot_idx = np.concatenate(rot_idx_list)
    digit_idx_t = torch.from_numpy(digit_idx)
    rot_idx_t = torch.from_numpy(rot_idx)

    n = len(all_images)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(all_images[train_idx], digit_idx_t[train_idx], rot_idx_t[train_idx])
    val_ds = TensorDataset(all_images[val_idx], digit_idx_t[val_idx], rot_idx_t[val_idx])
    train_dl = DataLoader(train_ds, batch_size=cnn_batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cnn_batch_size, shuffle=False)

    class DigitRotCNN(nn.Module):
        def __init__(self, embed_dim, n_digits, n_rots, dropout=0.3):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 64, 1, 1), robust to WHERE features land after rotation
            self.dropout = nn.Dropout(dropout)
            self.fc_embed = nn.Linear(64, embed_dim)
            self.embed_bn = nn.BatchNorm1d(embed_dim)
            self.fc_digit = nn.Linear(embed_dim, n_digits)
            self.fc_rot = nn.Linear(embed_dim, n_rots)

        def embed(self, x):
            x = Fnn.max_pool2d(Fnn.relu(self.bn1(self.conv1(x))), 2)   # 28->14
            x = Fnn.max_pool2d(Fnn.relu(self.bn2(self.conv2(x))), 2)   # 14->7
            x = Fnn.relu(self.bn3(self.conv3(x)))                     # 7->7
            x = self.global_pool(x).flatten(1)                        # (B, 64)
            x = self.dropout(x)
            e = self.embed_bn(self.fc_embed(x))
            return Fnn.normalize(e, dim=1) if l2_normalize_embed else e

        def forward(self, x):
            e = self.embed(x)
            return self.fc_digit(e), self.fc_rot(e)

    model = DigitRotCNN(embed_dim, n_digits, n_rots, dropout=cnn_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lr, weight_decay=cnn_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cnn_epochs)

    print(f"Training CNN ({cnn_epochs} epochs, embed_dim={embed_dim}, "
          f"rotation_aux_weight={rotation_aux_weight}, l2_normalize={l2_normalize_embed}, "
          f"dropout={cnn_dropout}, weight_decay={cnn_weight_decay})...")
    for epoch in range(cnn_epochs):
        model.train()
        total_loss, n_correct_digit, n_correct_rot, n_seen = 0.0, 0, 0, 0
        for xb, yb_digit, yb_rot in train_dl:
            xb, yb_digit, yb_rot = xb.to(device), yb_digit.to(device), yb_rot.to(device)
            optimizer.zero_grad()
            digit_logits, rot_logits = model(xb)
            digit_loss = Fnn.cross_entropy(digit_logits, yb_digit)
            rot_loss = Fnn.cross_entropy(rot_logits, yb_rot)
            loss = digit_loss + rotation_aux_weight * rot_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n_correct_digit += (digit_logits.argmax(-1) == yb_digit).sum().item()
            n_correct_rot += (rot_logits.argmax(-1) == yb_rot).sum().item()
            n_seen += len(xb)
        scheduler.step()

        model.eval()
        val_correct_digit, val_correct_rot, val_seen = 0, 0, 0
        with torch.no_grad():
            for xb, yb_digit, yb_rot in val_dl:
                xb, yb_digit, yb_rot = xb.to(device), yb_digit.to(device), yb_rot.to(device)
                digit_logits, rot_logits = model(xb)
                val_correct_digit += (digit_logits.argmax(-1) == yb_digit).sum().item()
                val_correct_rot += (rot_logits.argmax(-1) == yb_rot).sum().item()
                val_seen += len(xb)
        print(f"  epoch {epoch+1}/{cnn_epochs}: train_loss={total_loss/n_seen:.4f} "
              f"val_digit_acc={(val_correct_digit/val_seen if val_seen else float('nan')):.3f} "
              f"val_rotation_acc={(val_correct_rot/val_seen if val_seen else float('nan')):.3f}")

    model.eval()
    with torch.no_grad():
        codes = model.embed(all_images.to(device)).cpu().numpy().astype(np.float32)

    # ── Immediate silhouette report -- no separate visualize round trip needed ──
    try:
        from sklearn.metrics import silhouette_score
        print("\nEmbedding quality check (on the full generated set, before pairing):")
        print(f"  digit silhouette:    {silhouette_score(codes, digit_idx):.3f}")
        print(f"  rotation silhouette: {silhouette_score(codes, rot_idx):.3f}")
        print(f"  full-class (12-way) silhouette: {silhouette_score(codes, labels):.3f}")
        print("  (rough guide: <0.15 weak, 0.15-0.4 moderate, >0.4 strong separation)\n")
    except Exception as e:
        print(f"  (silhouette check skipped: {e})\n")

    out_perm = rng.permutation(len(codes))
    return codes[out_perm], labels[out_perm], class_names, digit_idx[out_perm], rot_idx[out_perm]


# ---------------------------------------------------------------------------
# Step 4: build pairs from EXACTLY the given rotation-pair combinations
# ---------------------------------------------------------------------------
def build_custom_rotation_pairs(codes, digit_idx, rot_idx, labels, digits, rotations,
                                 rotation_pairs, pairs_per_combo=5, seed=45):
    """For each digit, and for each (ra, rb) in rotation_pairs (given as
    actual rotation VALUES, e.g. degrees), draws `pairs_per_combo`
    anchor/partner pairs with anchor rotation ra and partner rotation rb.
    z = 1.0 if ra == rb, else 0.0. Only the listed combinations are
    produced -- nothing else.

    Returns:
        paired_codes: (M, 2, D) float32, M = n_digits * len(rotation_pairs) * pairs_per_combo
        paired_labels: (M,) int64 -- anchor's full class id
        pair_labels: (M,) float32
        partner_labels: (M,) int64 -- partner's full class id (for the sample printout)
    """
    rng = np.random.default_rng(seed)
    n_digits = len(digits)
    rot_value_to_idx = {r: i for i, r in enumerate(rotations)}

    for ra, rb in rotation_pairs:
        if ra not in rot_value_to_idx or rb not in rot_value_to_idx:
            raise ValueError(f"rotation_pairs entry ({ra}, {rb}) uses a value not in "
                              f"the inferred rotations list {rotations}.")

    by_digit_rot = {}
    for di in range(n_digits):
        for ri in range(len(rotations)):
            mask = (digit_idx == di) & (rot_idx == ri)
            by_digit_rot[(di, ri)] = np.where(mask)[0]
            if len(by_digit_rot[(di, ri)]) == 0:
                raise ValueError(f"No points found for digit index {di}, rotation index {ri}.")

    paired_codes, paired_labels, pair_labels, partner_labels = [], [], [], []

    for di in range(n_digits):
        for ra_val, rb_val in rotation_pairs:
            ra, rb = rot_value_to_idx[ra_val], rot_value_to_idx[rb_val]
            anchor_pool = by_digit_rot[(di, ra)]
            partner_pool = by_digit_rot[(di, rb)]
            z = 1.0 if ra == rb else 0.0

            for _ in range(pairs_per_combo):
                anchor_idx = int(rng.choice(anchor_pool))
                if ra == rb:
                    if len(partner_pool) > 1:
                        partner_idx = anchor_idx
                        while partner_idx == anchor_idx:
                            partner_idx = int(rng.choice(partner_pool))
                    else:
                        partner_idx = anchor_idx
                else:
                    partner_idx = int(rng.choice(partner_pool))

                paired_codes.append(np.stack([codes[anchor_idx], codes[partner_idx]], axis=0))
                paired_labels.append(labels[anchor_idx])
                pair_labels.append(z)
                partner_labels.append(labels[partner_idx])

    paired_codes = np.stack(paired_codes, axis=0).astype(np.float32)
    paired_labels = np.array(paired_labels, dtype=np.int64)
    pair_labels = np.array(pair_labels, dtype=np.float32)
    partner_labels = np.array(partner_labels, dtype=np.int64)

    perm = rng.permutation(len(paired_codes))
    return paired_codes[perm], paired_labels[perm], pair_labels[perm], partner_labels[perm]


# ---------------------------------------------------------------------------
# Step 5: save (paired -- for --dataset custom_pair; unpaired -- for --dataset custom)
# ---------------------------------------------------------------------------
def save_paired_dataset(out_dir, paired_codes, paired_labels, pair_labels, split="train"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.from_numpy(paired_codes), os.path.join(out_dir, f"{split}_data.pt"))
    torch.save(torch.from_numpy(paired_labels), os.path.join(out_dir, f"{split}_labels.pt"))
    torch.save(torch.from_numpy(pair_labels), os.path.join(out_dir, f"{split}_pair_labels.pt"))
    print(f"Saved PAIRED {split} split to {out_dir}: "
          f"{paired_codes.shape[0]} pairs "
          f"(positives: {int(pair_labels.sum())}/{len(pair_labels)})")


def save_unpaired_dataset(out_dir, codes, labels, split="train"):
    unpaired_dir = os.path.join(out_dir, "unpaired")
    os.makedirs(unpaired_dir, exist_ok=True)
    torch.save(torch.from_numpy(codes), os.path.join(unpaired_dir, f"{split}_data.pt"))
    torch.save(torch.from_numpy(labels), os.path.join(unpaired_dir, f"{split}_labels.pt"))
    print(f"Saved UNPAIRED {split} split to {unpaired_dir}: {len(codes)} points "
          f"(full dataset, no pairing subsetting)")


def save_metadata(out_dir, class_names, meta_extra=None):
    meta = {"n_classes": len(class_names), "class_names": class_names}
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Step 6: sample printout
# ---------------------------------------------------------------------------
def print_pair_sample(paired_labels, pair_labels, partner_labels, class_names, n_samples=20, seed=45):
    rng = np.random.default_rng(seed)
    n = len(paired_labels)
    sample_idx = rng.choice(n, size=min(n_samples, n), replace=False)

    print(f"\nSample of {len(sample_idx)} pairs (out of {n} total):")
    print(f"{'anchor':<10}{'partner':<10}{'z (same=1/diff=0)':<20}")
    print("-" * 40)
    for idx in sample_idx:
        anchor_name = class_names[paired_labels[idx]]
        partner_name = class_names[partner_labels[idx]]
        print(f"{anchor_name:<10}{partner_name:<10}{pair_labels[idx]:<20.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST rotation-pair dataset generator.")
    parser.add_argument("--digits", type=int, nargs="+", required=True, help="e.g. --digits 3 4 7")
    parser.add_argument("--rotation-pairs", type=str, required=True,
                         help="Comma-separated 'a-b' rotation VALUE pairs (degrees), e.g. "
                              "'0-90,90-180,180-270,270-0'. The set of rotations actually generated "
                              "is inferred as the sorted unique values appearing here.")
    parser.add_argument("--samples-per-class", type=int, default=1000, help="Images per (digit, rotation) combination")
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--pairs-per-combo", type=int, default=5, help="Pairs drawn per digit per rotation-pair combination")
    parser.add_argument("--cnn-epochs", type=int, default=15)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--cnn-dropout", type=float, default=0.3)
    parser.add_argument("--cnn-val-fraction", type=float, default=0.15)
    parser.add_argument("--no-l2-normalize", action="store_true",
                         help="Disable L2-normalizing the final embedding (normalized by default -- "
                              "keeps distances meaningful for downstream clustering).")
    parser.add_argument("--rotation-aux-weight", type=float, default=0.3,
                         help="Weight on the auxiliary rotation-prediction head. 0 = pure digit classifier "
                              "(becomes rotation-invariant, erases rotation structure). Default 0.3.")
    parser.add_argument("--mnist-root", type=str, default="./mnist_data")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    # parse rotation-pairs and infer the rotation set from it
    rotation_pairs = []
    for token in args.rotation_pairs.split(","):
        a, b = token.strip().split("-")
        rotation_pairs.append((int(a), int(b)))
    rotations = sorted(set(r for pair in rotation_pairs for r in pair))
    print(f"Inferred rotations from --rotation-pairs: {rotations}")

    codes, labels, class_names, digit_idx, rot_idx = generate_mnist_rotation_embeddings(
        digits=args.digits,
        rotations=rotations,
        samples_per_class=args.samples_per_class,
        embed_dim=args.embed_dim,
        seed=args.seed,
        mnist_root=args.mnist_root,
        cnn_epochs=args.cnn_epochs,
        cnn_lr=args.cnn_lr,
        cnn_weight_decay=args.cnn_weight_decay,
        cnn_batch_size=args.cnn_batch_size,
        cnn_dropout=args.cnn_dropout,
        val_fraction=args.cnn_val_fraction,
        rotation_aux_weight=args.rotation_aux_weight,
        l2_normalize_embed=not args.no_l2_normalize,
    )

    # full unpaired export first -- always, regardless of pairing subsetting
    save_unpaired_dataset(args.out_dir, codes, labels, split="train")

    paired_codes, paired_labels, pair_labels, partner_labels = build_custom_rotation_pairs(
        codes, digit_idx, rot_idx, labels,
        digits=args.digits, rotations=rotations,
        rotation_pairs=rotation_pairs,
        pairs_per_combo=args.pairs_per_combo,
        seed=args.seed,
    )
    save_paired_dataset(args.out_dir, paired_codes, paired_labels, pair_labels, split="train")

    save_metadata(
        args.out_dir, class_names,
        meta_extra={
            "digits": args.digits,
            "rotations": rotations,
            "rotation_pairs": rotation_pairs,
            "samples_per_class": args.samples_per_class,
            "embed_dim": args.embed_dim,
            "pairs_per_combo": args.pairs_per_combo,
            "cnn_epochs": args.cnn_epochs,
            "rotation_aux_weight": args.rotation_aux_weight,
            "seed": args.seed,
        },
    )

    print_pair_sample(paired_labels, pair_labels, partner_labels, class_names, n_samples=20, seed=args.seed)


if __name__ == "__main__":
    main()