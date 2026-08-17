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


def generate_embeddings_from_mnist_umap(
    digits,
    rotations,
    samples_per_class=200,
    embed_dim=10,
    seed=45,
    mnist_root="./mnist_data",
    umap_neighbors=15,
    umap_min_dist=0.1,
    supervise_by=None,
    umap_target_weight=0.5,
):
    """Real-data alternative to generate_embeddings(). Pulls actual MNIST
    digit images, applies each requested rotation, and fits UMAP on the
    raw pixel vectors (all classes together, one fit) to produce the final
    embedding.

    Why this matters (vs. the synthetic generator above): generate_embeddings
    assigns every digit and every rotation an arbitrary, mutually orthogonal
    random direction -- by construction, "6 rotated 180" and "9 rotated 0"
    are just as separable as any other pair, because nothing in that
    generator ever looks at what a 6 or a 9 actually looks like. Here,
    since UMAP is fit on real rotated pixel data, any genuine visual
    ambiguity (rotated 6 vs 9, near-rotation-invariant digits like 0/1/8,
    etc.) will show up as real proximity/overlap in the resulting
    embedding space -- exactly the effect you can't get from the
    synthetic generator.

    Fitting UMAP ONCE across all requested (digit, rotation) combinations
    together (rather than per-class) is what lets true cross-class
    similarity surface -- if you fit UMAP separately per class, you
    throw away exactly the information you're trying to preserve.

    supervise_by: None (default), "digit", or "class".
        Fully unsupervised UMAP on raw pixels can fail to separate even
        digit identity if samples_per_class is small or the pixel-space
        neighborhood structure is noisy -- and when that happens, NO
        clustering method downstream (with or without pairwise
        supervision) has anything real to find, since the bottleneck is
        the embedding itself, not the clustering algorithm.
        - "digit": semi-supervised UMAP, nudged (via `umap_target_weight`)
          to respect digit identity while rotation structure is left to
          emerge from the real pixel data on its own. This guarantees
          digit-level separability (so DeepDPM has real structure to
          discover) while still allowing genuine rotation-driven overlap
          -- e.g. 6@180 vs 9@0 -- to appear where it actually exists in
          the pixels, rather than being fabricated OR accidentally
          destroyed by an unstable unsupervised fit.
        - "class": semi-supervised on the full (digit, rotation) class
          instead -- guarantees full separability, useful as a sanity
          baseline but removes the rotation-ambiguity phenomenon you're
          trying to study.
        - None: fully unsupervised (original behavior) -- most faithful
          to "let the real data speak for itself," but can fail to
          separate even digits if UMAP's fit is unstable on this data.
    umap_target_weight: 0.0-1.0, only used when supervise_by is set.
        0 = ignore labels entirely (same as unsupervised); 1 = weight
        labels very heavily (approaches "class" behavior even under
        "digit" supervision). 0.5 is a reasonable middle ground: real
        pixel structure still dominates, labels just anchor digit
        identity so the fit doesn't collapse it by chance.

    Returns the same 5-tuple as generate_embeddings(), so every pairing
    function (build_pairs, build_pairs_same_digit,
    build_pairs_same_digit_exhaustive, build_pairs_same_digit_custom)
    works unchanged on top of it.

    Requires `torchvision` (for the MNIST download) and `umap-learn`.
    NOTE: downloading MNIST requires unrestricted internet access to
    reach torchvision's hosting mirrors -- this will fail in network-
    sandboxed environments and needs to be run somewhere with normal
    internet access (e.g. Colab).
    """
    try:
        import torchvision
        import torchvision.transforms.functional as TF
    except ImportError as e:
        raise ImportError(
            "generate_embeddings_from_mnist_umap requires torchvision. "
            "Install with: pip install torchvision"
        ) from e
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "generate_embeddings_from_mnist_umap requires umap-learn. "
            "Install with: pip install umap-learn"
        ) from e

    rng = np.random.default_rng(seed)

    print(f"Loading MNIST (root={mnist_root}, download if needed)...")
    dataset = torchvision.datasets.MNIST(root=mnist_root, train=True, download=True)

    n_rots = len(rotations)
    all_pixel_vectors = []
    labels = []
    digit_idx_list = []
    rot_idx_list = []
    class_names = []

    class_idx = 0
    for di, digit in enumerate(digits):
        pool_mask = dataset.targets == digit
        pool = dataset.data[pool_mask]  # (n_available, 28, 28) uint8
        if len(pool) < samples_per_class:
            raise ValueError(
                f"Not enough MNIST images for digit {digit}: requested "
                f"{samples_per_class}, only {len(pool)} available."
            )
        chosen = rng.choice(len(pool), size=samples_per_class, replace=False)
        base_imgs = pool[chosen].float()  # (samples_per_class, 28, 28)

        for ri, rot in enumerate(rotations):
            rotated = torch.stack([
                TF.rotate(img.unsqueeze(0), angle=float(rot)).squeeze(0)
                for img in base_imgs
            ])
            flat = (rotated.view(samples_per_class, -1) / 255.0).numpy()
            all_pixel_vectors.append(flat)
            labels.append(np.full(samples_per_class, class_idx, dtype=np.int64))
            digit_idx_list.append(np.full(samples_per_class, di, dtype=np.int64))
            rot_idx_list.append(np.full(samples_per_class, ri, dtype=np.int64))
            class_names.append(f"{digit}_{ri}")
            class_idx += 1

    all_pixel_vectors = np.concatenate(all_pixel_vectors, axis=0)  # (N, 784)
    labels = np.concatenate(labels)
    digit_idx = np.concatenate(digit_idx_list)
    rot_idx = np.concatenate(rot_idx_list)

    print(f"Fitting UMAP (n_components={embed_dim}, n_neighbors={umap_neighbors}, "
          f"min_dist={umap_min_dist}, supervise_by={supervise_by}) "
          f"on {len(all_pixel_vectors)} rotated MNIST images...")
    reducer = umap.UMAP(
        n_components=embed_dim, n_neighbors=umap_neighbors, min_dist=umap_min_dist,
        random_state=seed,
        target_weight=umap_target_weight if supervise_by else 1.0,
    )

    if supervise_by == "digit":
        codes = reducer.fit_transform(all_pixel_vectors, y=digit_idx).astype(np.float32)
    elif supervise_by == "class":
        codes = reducer.fit_transform(all_pixel_vectors, y=labels).astype(np.float32)
    elif supervise_by is not None:
        raise ValueError(f"supervise_by must be None, 'digit', or 'class', got {supervise_by!r}")
    else:
        codes = reducer.fit_transform(all_pixel_vectors).astype(np.float32)

    perm = rng.permutation(len(codes))
    return codes[perm], labels[perm], class_names, digit_idx[perm], rot_idx[perm]


# ---------------------------------------------------------------------------
# Pair construction -- materializes actual (anchor, partner, z) triples
# ---------------------------------------------------------------------------
def generate_embeddings_from_mnist_cnn(
    digits,
    rotations,
    samples_per_class=200,
    embed_dim=10,
    seed=45,
    mnist_root="./mnist_data",
    cnn_epochs=8,
    cnn_lr=1e-3,
    cnn_batch_size=64,
    val_fraction=0.15,
    rotation_aux_weight=0.3,
    device=None,
):
    """Real-data alternative to generate_embeddings() / generate_embeddings_from_mnist_umap().

    Rather than reducing raw pixels with UMAP (which is dominated by
    rotation angle, not digit identity -- rotating an image moves pixel
    mass more than changing the digit does, so raw-pixel UMAP can fail to
    separate even easy digit sets under rotation), this trains a small CNN
    classifier to predict DIGIT identity from the rotated images, then uses
    its learned penultimate-layer activations as the embedding.

    Why this is the right fix: the CNN's features are *trained* to be
    digit-separable (that's its objective), so digit-level clustering has
    real structure to find, by construction. Any genuine rotation-driven
    ambiguity (e.g. 6 vs 9 near 180 degrees) shows up NATURALLY -- as
    places where the classifier itself gets confused -- rather than being
    fabricated (synthetic generator) or erased/imposed by a UMAP
    hyperparameter (mnist-umap generator).

    IMPORTANT -- rotation_aux_weight: training on digit identity ALONE
    teaches the network to become rotation-INVARIANT -- that's the whole
    point of a well-trained digit classifier, and it actively discards
    rotation information as nuisance variation. Empirically this can drive
    rotation silhouette to ~0, i.e. no rotation structure left to find at
    all, regardless of any downstream pairwise supervision. Setting
    rotation_aux_weight > 0 adds a second auxiliary head that also
    predicts rotation from the same shared embedding, trained jointly
    (loss = digit_loss + rotation_aux_weight * rotation_loss). This keeps
    the embedding predictive of BOTH digit (dominant factor) and rotation
    (finer, auxiliary factor) -- mirroring what digit_sep/rot_sep did
    explicitly in the synthetic generator, but now grounded in real MNIST
    pixels via an actually-trained encoder. 0.3 is a reasonable default;
    raise it if rotation structure still isn't showing up, lower it if
    digit separation degrades.

    Trains on a train/val split of the SAME rotated-image pool used to
    build the final dataset (this is meant to produce a good embedding
    space for a clustering demonstration, not a held-out generalization
    benchmark -- if you need a strict train/test separation for the
    embedding model itself, extend this to pull the CNN's training images
    from a disjoint MNIST split).

    Returns the same 5-tuple as generate_embeddings(), so every pairing
    function works unchanged on top of it.

    Requires `torch` + `torchvision` and internet access to download MNIST.
    """
    try:
        import torchvision
        import torchvision.transforms.functional as TF
    except ImportError as e:
        raise ImportError("generate_embeddings_from_mnist_cnn requires torchvision.") from e
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

    all_images = []   # (N, 1, 28, 28) float in [0,1]
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
            all_images.append((rotated / 255.0).unsqueeze(1))  # (samples_per_class, 1, 28, 28)
            labels.append(np.full(samples_per_class, class_idx, dtype=np.int64))
            digit_idx_list.append(np.full(samples_per_class, di, dtype=np.int64))
            rot_idx_list.append(np.full(samples_per_class, ri, dtype=np.int64))
            class_names.append(f"{digit}_{ri}")
            class_idx += 1

    all_images = torch.cat(all_images, dim=0)                 # (N, 1, 28, 28)
    labels = np.concatenate(labels)
    digit_idx = np.concatenate(digit_idx_list)
    rot_idx = np.concatenate(rot_idx_list)
    digit_idx_t = torch.from_numpy(digit_idx)
    rot_idx_t = torch.from_numpy(rot_idx)

    # train/val split for early-stopping sanity (not a strict generalization test)
    n = len(all_images)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(all_images[train_idx], digit_idx_t[train_idx], rot_idx_t[train_idx])
    val_ds = TensorDataset(all_images[val_idx], digit_idx_t[val_idx], rot_idx_t[val_idx])
    train_dl = DataLoader(train_ds, batch_size=cnn_batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cnn_batch_size, shuffle=False)

    class DigitCNN(nn.Module):
        def __init__(self, embed_dim, n_digits, n_rots):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc_embed = nn.Linear(32 * 7 * 7, embed_dim)
            self.fc_digit = nn.Linear(embed_dim, n_digits)
            self.fc_rot = nn.Linear(embed_dim, n_rots)  # auxiliary head -- keeps rotation info in the shared embedding

        def embed(self, x):
            x = Fnn.max_pool2d(Fnn.relu(self.conv1(x)), 2)   # 28->14
            x = Fnn.max_pool2d(Fnn.relu(self.conv2(x)), 2)   # 14->7
            x = x.view(x.size(0), -1)
            return Fnn.relu(self.fc_embed(x))

        def forward(self, x):
            e = self.embed(x)
            return self.fc_digit(e), self.fc_rot(e)

    model = DigitCNN(embed_dim, n_digits, n_rots).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lr)

    print(f"Training digit-classification CNN ({cnn_epochs} epochs, embed_dim={embed_dim}, "
          f"rotation_aux_weight={rotation_aux_weight}) to produce a trained, digit-separable "
          f"embedding space that retains rotation-relevant structure...")
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
              f"train_digit_acc={n_correct_digit/n_seen:.3f} "
              f"val_digit_acc={(val_correct_digit/val_seen if val_seen else float('nan')):.3f} "
              f"val_rotation_acc={(val_correct_rot/val_seen if val_seen else float('nan')):.3f}")

    # extract embeddings for ALL images (train + val), in original order
    model.eval()
    with torch.no_grad():
        codes = model.embed(all_images.to(device)).cpu().numpy().astype(np.float32)

    out_perm = rng.permutation(len(codes))
    return codes[out_perm], labels[out_perm], class_names, digit_idx[out_perm], rot_idx[out_perm]



    """For each point in `codes`, build `pairs_per_sample` (anchor, partner)
    pairs, each independently drawn as positive (same class) with
    probability `positive_prob`, else negative (different class).

    Returns:
        paired_codes: (M, 2, D) float32 array, M = N * pairs_per_sample
        paired_labels: (M,) int64 array -- the anchor's class
        pair_labels: (M,) float32 array -- 1.0 if same class, 0.0 if different
        partner_labels: (M,) int64 array -- the partner's class (for inspection/printing only)
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
    partner_labels = []

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
            partner_labels.append(labels[partner_idx])

    paired_codes = np.stack(paired_codes, axis=0).astype(np.float32)   # (M, 2, D)
    paired_labels = np.array(paired_labels, dtype=np.int64)            # (M,)
    pair_labels = np.array(pair_labels, dtype=np.float32)              # (M,)
    partner_labels = np.array(partner_labels, dtype=np.int64)          # (M,)

    # shuffle across the whole set so repeated-anchor pairs aren't adjacent
    perm = rng.permutation(len(paired_codes))
    return paired_codes[perm], paired_labels[perm], pair_labels[perm], partner_labels[perm]


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
        partner_labels: (M,) int64 -- partner's full class id (for inspection/printing only)
    """
    rng = np.random.default_rng(seed)
    n = len(codes)

    by_digit = {}
    for d in np.unique(digit_idx):
        by_digit[d] = np.where(digit_idx == d)[0]

    paired_codes = []
    paired_labels = []
    pair_labels = []
    partner_labels = []

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
            partner_labels.append(labels[partner_idx])

    paired_codes = np.stack(paired_codes, axis=0).astype(np.float32)
    paired_labels = np.array(paired_labels, dtype=np.int64)
    pair_labels = np.array(pair_labels, dtype=np.float32)
    partner_labels = np.array(partner_labels, dtype=np.int64)

    perm = rng.permutation(len(paired_codes))
    return paired_codes[perm], paired_labels[perm], pair_labels[perm], partner_labels[perm]


def build_pairs_same_digit_exhaustive(codes, digit_idx, rot_idx, labels, digits, rotations,
                                       pairs_per_combo=5, seed=45):
    """Exhaustive version of build_pairs_same_digit: guarantees every
    ordered (digit, rotation_a, rotation_b) combination is represented,
    including same-rotation combos (rotation_a == rotation_b, z=1) and
    every cross-rotation combo (z=0) -- e.g. for rotations [0,90,180,270]
    this covers all 4x4=16 (rotation_a, rotation_b) pairs per digit:
    0-0, 0-90, 0-180, 0-270, 90-0, 90-90, ..., 270-180, 270-270.

    Unlike build_pairs_same_digit (which independently samples ONE random
    partner per anchor point and can, by chance, miss rare combinations
    entirely -- confirmed empirically at small sample sizes), this
    explicitly loops over every combination and draws `pairs_per_combo`
    anchor/partner pairs for each, so coverage is guaranteed rather than
    probabilistic.

    Args:
        digits, rotations: the same lists passed to generate_embeddings
        pairs_per_combo: how many (anchor, partner) pairs to draw for each
            of the n_digits * n_rotations^2 combinations

    Returns:
        paired_codes: (M, 2, D) float32, M = n_digits * n_rotations^2 * pairs_per_combo
        paired_labels: (M,) int64 -- anchor's full class id
        pair_labels: (M,) float32 -- 1.0 same rotation, 0.0 different rotation
        partner_labels: (M,) int64 -- partner's full class id
    """
    rng = np.random.default_rng(seed)
    n_digits = len(digits)
    n_rots = len(rotations)

    # index points by (digit_idx, rot_idx) for direct lookup
    by_digit_rot = {}
    for di in range(n_digits):
        for ri in range(n_rots):
            mask = (digit_idx == di) & (rot_idx == ri)
            by_digit_rot[(di, ri)] = np.where(mask)[0]
            if len(by_digit_rot[(di, ri)]) == 0:
                raise ValueError(f"No points found for digit index {di}, rotation index {ri} -- "
                                  f"check that samples_per_class > 0 for every (digit, rotation) combo.")

    paired_codes = []
    paired_labels = []
    pair_labels = []
    partner_labels = []

    for di in range(n_digits):
        for ra in range(n_rots):
            anchor_pool = by_digit_rot[(di, ra)]
            for rb in range(n_rots):
                partner_pool = by_digit_rot[(di, rb)]
                z = 1.0 if ra == rb else 0.0

                for _ in range(pairs_per_combo):
                    anchor_idx = int(rng.choice(anchor_pool))
                    if ra == rb:
                        # avoid trivially pairing a point with itself when possible
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


def build_pairs_same_digit_custom(codes, digit_idx, rot_idx, labels, digits, rotations,
                                   rotation_pairs, pairs_per_combo=5, seed=45):
    """Same-digit pairing restricted to an explicit, user-specified list of
    (rotation_a, rotation_b) combinations -- e.g. only the cyclic
    transitions [(0,90), (90,180), (180,270), (270,0)], rather than every
    possible combination.

    For each digit, and for each (ra, rb) pair in `rotation_pairs` (given
    as actual rotation VALUES, e.g. degrees -- matching the `rotations`
    list, not indices), draws `pairs_per_combo` anchor/partner pairs with
    anchor rotation ra and partner rotation rb. z = 1.0 if ra == rb (only
    relevant if you explicitly include a same-rotation pair like (0, 0)),
    else 0.0.

    Args:
        digits, rotations: the same lists passed to generate_embeddings
        rotation_pairs: list of (ra, rb) tuples using actual rotation
            values from `rotations`, e.g. [(0, 90), (90, 180), (180, 270), (270, 0)]
        pairs_per_combo: how many pairs to draw per digit per combination

    Returns:
        paired_codes: (M, 2, D) float32, M = n_digits * len(rotation_pairs) * pairs_per_combo
        paired_labels: (M,) int64 -- anchor's full class id
        pair_labels: (M,) float32 -- 1.0 if ra==rb else 0.0
        partner_labels: (M,) int64 -- partner's full class id
    """
    rng = np.random.default_rng(seed)
    n_digits = len(digits)
    rot_value_to_idx = {r: i for i, r in enumerate(rotations)}

    for ra, rb in rotation_pairs:
        if ra not in rot_value_to_idx or rb not in rot_value_to_idx:
            raise ValueError(
                f"rotation_pairs entry ({ra}, {rb}) uses a value not in `rotations` "
                f"({rotations}). Use actual rotation values (e.g. degrees), not indices."
            )

    by_digit_rot = {}
    for di in range(n_digits):
        for ri in range(len(rotations)):
            mask = (digit_idx == di) & (rot_idx == ri)
            by_digit_rot[(di, ri)] = np.where(mask)[0]
            if len(by_digit_rot[(di, ri)]) == 0:
                raise ValueError(f"No points found for digit index {di}, rotation index {ri}.")

    paired_codes = []
    paired_labels = []
    pair_labels = []
    partner_labels = []

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


def print_pair_sample(paired_labels, pair_labels, partner_labels, class_names, n_samples=20, seed=45):
    """Prints a random sample of pairs (anchor label, partner label, z) so
    you can eyeball whether the pairing logic is doing what you expect --
    e.g. that 'same-digit' mode never pairs across digits, or that the
    same/different balance looks reasonable."""
    rng = np.random.default_rng(seed)
    n = len(paired_labels)
    sample_idx = rng.choice(n, size=min(n_samples, n), replace=False)

    print(f"\nSample of {len(sample_idx)} pairs (out of {n} total):")
    print(f"{'anchor':<10}{'partner':<10}{'z (same=1/diff=0)':<20}")
    print("-" * 40)
    for idx in sample_idx:
        anchor_name = class_names[paired_labels[idx]]
        partner_name = class_names[partner_labels[idx]]
        z = pair_labels[idx]
        print(f"{anchor_name:<10}{partner_name:<10}{z:<20.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ROT_PAIR digit+rotation embeddings (.pt layout).")
    parser.add_argument("--digits", type=int, nargs="+", default=[3, 4, 7])
    parser.add_argument("--rotations", type=int, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--data-source", type=str, default="synthetic", choices=["synthetic", "mnist-umap", "mnist-cnn"],
                         help="'synthetic': arbitrary orthogonal directions per digit/rotation -- guaranteed separable, "
                              "cannot represent real visual ambiguity (e.g. 6@180 vs 9@0). "
                              "'mnist-umap': real MNIST digit images, rotated, embedded via a single UMAP fit across all "
                              "classes -- preserves genuine pixel-level structure, but raw-pixel distance can be "
                              "dominated by rotation angle rather than digit identity, sometimes failing to separate "
                              "even digits. "
                              "'mnist-cnn': trains a small CNN classifier on digit identity from the rotated images and "
                              "uses its learned features as the embedding -- guarantees digit-level separability by "
                              "construction (trained for it), while genuine rotation-driven ambiguity (e.g. 6 vs 9) "
                              "shows up naturally wherever the classifier itself struggles. Recommended over "
                              "mnist-umap if digit separation isn't holding up. "
                              "Both mnist-* modes require torchvision + internet access to download MNIST.")
    parser.add_argument("--digit-sep", type=float, default=8.0, help="[synthetic only]")
    parser.add_argument("--rot-sep", type=float, default=4.0, help="[synthetic only]")
    parser.add_argument("--noise-std", type=float, default=1.0, help="[synthetic only]")
    parser.add_argument("--mnist-root", type=str, default="./mnist_data", help="[mnist-umap only] download/cache dir for MNIST")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="[mnist-umap only] UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="[mnist-umap only] UMAP min_dist")
    parser.add_argument("--supervise-by", type=str, default=None, choices=[None, "digit", "class"],
                         help="[mnist-umap only] None: fully unsupervised (can fail to separate even digits if the fit "
                              "is unstable). 'digit': semi-supervised to guarantee digit-level separability while "
                              "leaving rotation structure to emerge naturally from real pixels -- recommended if you "
                              "want genuine rotation-ambiguity (e.g. 6@180 vs 9@0) without risking a collapsed fit. "
                              "'class': semi-supervised on full (digit,rotation) classes -- guarantees full separability "
                              "but removes the rotation-ambiguity phenomenon.")
    parser.add_argument("--umap-target-weight", type=float, default=0.5,
                         help="[mnist-umap only, requires --supervise-by] 0=ignore labels, 1=weight labels heavily. Default 0.5.")
    parser.add_argument("--cnn-epochs", type=int, default=8, help="[mnist-cnn only] training epochs for the digit-classification CNN")
    parser.add_argument("--cnn-lr", type=float, default=1e-3, help="[mnist-cnn only]")
    parser.add_argument("--cnn-batch-size", type=int, default=64, help="[mnist-cnn only]")
    parser.add_argument("--cnn-val-fraction", type=float, default=0.15, help="[mnist-cnn only] held-out fraction for monitoring training (not a strict generalization test)")
    parser.add_argument("--rotation-aux-weight", type=float, default=0.3,
                         help="[mnist-cnn only] weight on an auxiliary rotation-prediction head, trained jointly with "
                              "the main digit head. 0 = pure digit classifier (will become rotation-invariant, likely "
                              "erasing rotation structure entirely). >0 keeps rotation-relevant information in the "
                              "shared embedding. Default 0.3.")
    parser.add_argument("--pairs-per-sample", type=int, default=1, help="How many (anchor, partner) pairs to generate per base point (used by 'random'/'same-digit' modes; ignored by 'same-digit-exhaustive')")
    parser.add_argument("--positive-prob", type=float, default=0.5, help="Probability a generated pair is 'positive' under the chosen --pairing-mode (used by 'random'/'same-digit' modes; ignored by 'same-digit-exhaustive')")
    parser.add_argument("--pairs-per-combo", type=int, default=5, help="Pairs to draw per combination -- used by 'same-digit-exhaustive' and 'same-digit-custom' modes")
    parser.add_argument("--rotation-pairs", type=str, default=None,
                         help="Comma-separated list of 'a-b' rotation VALUE pairs (matching --rotations, e.g. degrees) "
                              "to restrict pairing to -- e.g. '0-90,90-180,180-270,270-0' for a cyclic pattern. "
                              "Required when --pairing-mode same-digit-custom.")
    parser.add_argument("--pairing-mode", type=str, default="random",
                         choices=["random", "same-digit", "same-digit-exhaustive", "same-digit-custom"],
                         help="'random': pairs from anywhere (z=1 same full class, z=0 different class -- redundant with what unsupervised clustering already finds easily). "
                              "'same-digit': pairs restricted to the same digit, one random partner per anchor point (z=1 same rotation, z=0 different rotation) -- coverage of specific rotation combinations is NOT guaranteed, especially at small sample sizes. "
                              "'same-digit-exhaustive': same restriction, but guarantees every (digit, rotation_a, rotation_b) combination appears at least --pairs-per-combo times. "
                              "'same-digit-custom': same restriction, but only the specific (rotation_a, rotation_b) combinations given in --rotation-pairs (e.g. a cyclic subset instead of the full grid).")
    parser.add_argument("--test-fraction", type=float, default=0.0, help="Fraction of base points held out for a test split (0 = no test split, matching your original runs' 'Test data not found' behavior)")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    if args.data_source == "mnist-umap":
        codes, labels, class_names, digit_idx, rot_idx = generate_embeddings_from_mnist_umap(
            digits=args.digits,
            rotations=args.rotations,
            samples_per_class=args.samples_per_class,
            embed_dim=args.embed_dim,
            seed=args.seed,
            mnist_root=args.mnist_root,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            supervise_by=args.supervise_by,
            umap_target_weight=args.umap_target_weight,
        )
    elif args.data_source == "mnist-cnn":
        codes, labels, class_names, digit_idx, rot_idx = generate_embeddings_from_mnist_cnn(
            digits=args.digits,
            rotations=args.rotations,
            samples_per_class=args.samples_per_class,
            embed_dim=args.embed_dim,
            seed=args.seed,
            mnist_root=args.mnist_root,
            cnn_epochs=args.cnn_epochs,
            cnn_lr=args.cnn_lr,
            cnn_batch_size=args.cnn_batch_size,
            val_fraction=args.cnn_val_fraction,
            rotation_aux_weight=args.rotation_aux_weight,
        )
    else:
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

    parsed_rotation_pairs = None
    if args.rotation_pairs:
        parsed_rotation_pairs = []
        for token in args.rotation_pairs.split(","):
            a, b = token.strip().split("-")
            parsed_rotation_pairs.append((int(a), int(b)))

    if args.pairing_mode == "same-digit-custom" and not parsed_rotation_pairs:
        raise ValueError("--pairing-mode same-digit-custom requires --rotation-pairs, e.g. '0-90,90-180,180-270,270-0'")

    def _build(codes_, labels_, digit_idx_, rot_idx_, seed_):
        if args.pairing_mode == "same-digit-custom":
            return build_pairs_same_digit_custom(
                codes_, digit_idx_, rot_idx_, labels_,
                digits=args.digits, rotations=args.rotations,
                rotation_pairs=parsed_rotation_pairs,
                pairs_per_combo=args.pairs_per_combo,
                seed=seed_,
            )
        if args.pairing_mode == "same-digit-exhaustive":
            return build_pairs_same_digit_exhaustive(
                codes_, digit_idx_, rot_idx_, labels_,
                digits=args.digits, rotations=args.rotations,
                pairs_per_combo=args.pairs_per_combo,
                seed=seed_,
            )
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

    train_paired_codes, train_paired_labels, train_pair_labels, train_partner_labels = _build(
        train_codes, train_labels, train_digit_idx, train_rot_idx, args.seed
    )
    save_pt_dataset(args.out_dir, train_paired_codes, train_paired_labels, train_pair_labels, split="train")

    if test_codes is not None and len(test_codes) > 0:
        test_paired_codes, test_paired_labels, test_pair_labels, test_partner_labels = _build(
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
            "data_source": args.data_source,
            "pairing_mode": args.pairing_mode,
            "rotation_pairs": parsed_rotation_pairs,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
        },
    )

    # ── Print a sample of pairs for a quick sanity check ────────────────
    print_pair_sample(train_paired_labels, train_pair_labels, train_partner_labels, class_names, n_samples=20, seed=args.seed)


if __name__ == "__main__":
    main()