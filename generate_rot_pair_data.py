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
import sys

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
    full_class_weight=1.0,
    only_full_class_head=False,
    l2_normalize_embed=True,
    device=None,
):
    """Downloads MNIST, builds rotated images for every (digit, rotation)
    combination, trains a two-head CNN (digit + rotation), and returns the
    penultimate-layer embedding for every image.

    samples_per_class: either a single int (broadcast to every digit, same
    count for all) OR a list of ints, one per digit in `digits`, e.g.
    digits=[3,4,7], samples_per_class=[500, 800, 300] pulls 500 images of
    digit 3, 800 of digit 4, 300 of digit 7 -- each still expanded across
    every requested rotation, so digit 3 contributes 500 * len(rotations)
    total images, digit 4 contributes 800 * len(rotations), etc.

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

    full_class_weight: adds a THIRD head trained directly on the full
    joint (digit, rotation) class label -- i.e. each of the n_digits *
    n_rotations combinations as its own distinct training target, rather
    than only supervising digit identity and rotation angle separately as
    two marginal signals. This is a more direct way to push the embedding
    to separate every individual cluster, since the digit+rotation heads
    only combine into cluster-level separation indirectly (as a product of
    two marginal objectives), whereas this head's loss directly penalizes
    any confusion between two specific (digit, rotation) combinations.
    Default 1.0 (on). Set to 0 to disable and train with only the digit +
    rotation marginal heads (the original behavior).

    only_full_class_head: if True, REMOVES the separate digit and rotation
    heads entirely -- the model has only the full-class head, trained
    purely on the joint (digit, rotation) label, and rotation_aux_weight /
    full_class_weight are ignored (loss is just cross-entropy on the full
    class). digit and rotation accuracy are still reported, but derived
    POST-HOC by decoding the predicted full class back into its
    (digit, rotation) components (class_idx = digit_i * n_rots + rot_i,
    so digit_pred = pred // n_rots, rot_pred = pred % n_rots) -- these are
    informational only, never used in the loss.

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

    # Resolve samples_per_class to one count per digit -- accepts a single
    # int (broadcast to all digits) or a list matching len(digits).
    if isinstance(samples_per_class, (list, tuple)):
        if len(samples_per_class) != n_digits:
            raise ValueError(f"samples_per_class list has {len(samples_per_class)} entries "
                              f"but there are {n_digits} digits -- must match, or pass a single int.")
        samples_per_digit = list(samples_per_class)
    else:
        samples_per_digit = [samples_per_class] * n_digits
    print(f"Samples per digit: {dict(zip(digits, samples_per_digit))}")

    all_images = []
    labels = []
    digit_idx_list = []
    rot_idx_list = []
    class_names = []

    class_idx = 0
    for di, digit in enumerate(digits):
        n_samples_this_digit = samples_per_digit[di]
        pool_mask = dataset.targets == digit
        pool = dataset.data[pool_mask]
        if len(pool) < n_samples_this_digit:
            raise ValueError(f"Not enough MNIST images for digit {digit}: requested "
                              f"{n_samples_this_digit}, only {len(pool)} available.")
        chosen = rng.choice(len(pool), size=n_samples_this_digit, replace=False)
        base_imgs = pool[chosen].float()

        for ri, rot in enumerate(rotations):
            rotated = torch.stack([
                TF.rotate(img.unsqueeze(0), angle=float(rot)).squeeze(0)
                for img in base_imgs
            ])
            all_images.append((rotated / 255.0).unsqueeze(1))
            labels.append(np.full(n_samples_this_digit, class_idx, dtype=np.int64))
            digit_idx_list.append(np.full(n_samples_this_digit, di, dtype=np.int64))
            rot_idx_list.append(np.full(n_samples_this_digit, ri, dtype=np.int64))
            class_names.append(f"{digit}_{ri}")
            class_idx += 1

    all_images = torch.cat(all_images, dim=0)
    labels = np.concatenate(labels)
    digit_idx = np.concatenate(digit_idx_list)
    rot_idx = np.concatenate(rot_idx_list)
    digit_idx_t = torch.from_numpy(digit_idx)
    rot_idx_t = torch.from_numpy(rot_idx)
    full_class_t = torch.from_numpy(labels)   # the joint (digit, rotation) class id -- one label per cluster

    n = len(all_images)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(all_images[train_idx], digit_idx_t[train_idx], rot_idx_t[train_idx], full_class_t[train_idx])
    val_ds = TensorDataset(all_images[val_idx], digit_idx_t[val_idx], rot_idx_t[val_idx], full_class_t[val_idx])
    train_dl = DataLoader(train_ds, batch_size=cnn_batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cnn_batch_size, shuffle=False)

    class DigitRotCNN(nn.Module):
        def __init__(self, embed_dim, n_digits, n_rots, dropout=0.3, only_full_class_head=False):
            super().__init__()
            self.only_full_class_head = only_full_class_head
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
            self.fc_full_class = nn.Linear(embed_dim, n_digits * n_rots)  # predicts the FULL joint class directly
            if not only_full_class_head:
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
            if self.only_full_class_head:
                return self.fc_full_class(e)
            return self.fc_digit(e), self.fc_rot(e), self.fc_full_class(e)

    model = DigitRotCNN(embed_dim, n_digits, n_rots, dropout=cnn_dropout,
                         only_full_class_head=only_full_class_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lr, weight_decay=cnn_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cnn_epochs)

    print(f"Training CNN ({cnn_epochs} epochs, embed_dim={embed_dim}, "
          f"only_full_class_head={only_full_class_head}, "
          f"rotation_aux_weight={rotation_aux_weight if not only_full_class_head else 'n/a'}, "
          f"full_class_weight={full_class_weight if not only_full_class_head else 'n/a (sole head)'}, "
          f"l2_normalize={l2_normalize_embed}, dropout={cnn_dropout}, weight_decay={cnn_weight_decay})...")
    for epoch in range(cnn_epochs):
        model.train()
        total_loss, n_correct_digit, n_correct_rot, n_correct_full, n_seen = 0.0, 0, 0, 0, 0
        for xb, yb_digit, yb_rot, yb_full in train_dl:
            xb, yb_digit, yb_rot, yb_full = xb.to(device), yb_digit.to(device), yb_rot.to(device), yb_full.to(device)
            optimizer.zero_grad()
            if only_full_class_head:
                full_logits = model(xb)
                loss = Fnn.cross_entropy(full_logits, yb_full)
                pred_full = full_logits.argmax(-1)
                # digit/rotation "accuracy" here is derived POST-HOC by decoding
                # the full-class prediction (class_idx = digit_i * n_rots + rot_i)
                # -- informational only, never used in the loss.
                pred_digit_derived = pred_full // n_rots
                pred_rot_derived = pred_full % n_rots
                n_correct_digit += (pred_digit_derived == yb_digit).sum().item()
                n_correct_rot += (pred_rot_derived == yb_rot).sum().item()
            else:
                digit_logits, rot_logits, full_logits = model(xb)
                digit_loss = Fnn.cross_entropy(digit_logits, yb_digit)
                rot_loss = Fnn.cross_entropy(rot_logits, yb_rot)
                loss = digit_loss + rotation_aux_weight * rot_loss
                if full_class_weight > 0:
                    full_loss = Fnn.cross_entropy(full_logits, yb_full)
                    loss = loss + full_class_weight * full_loss
                n_correct_digit += (digit_logits.argmax(-1) == yb_digit).sum().item()
                n_correct_rot += (rot_logits.argmax(-1) == yb_rot).sum().item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n_correct_full += (full_logits.argmax(-1) == yb_full).sum().item()
            n_seen += len(xb)
        scheduler.step()

        model.eval()
        val_correct_digit, val_correct_rot, val_correct_full, val_seen = 0, 0, 0, 0
        val_digit_preds, val_digit_true = [], []
        val_full_preds, val_full_true = [], []
        with torch.no_grad():
            for xb, yb_digit, yb_rot, yb_full in val_dl:
                xb, yb_digit, yb_rot, yb_full = xb.to(device), yb_digit.to(device), yb_rot.to(device), yb_full.to(device)
                if only_full_class_head:
                    full_logits = model(xb)
                    pred_full = full_logits.argmax(-1)
                    pred_digit = pred_full // n_rots           # derived, not directly predicted
                    val_correct_rot += (pred_full % n_rots == yb_rot).sum().item()
                else:
                    digit_logits, rot_logits, full_logits = model(xb)
                    pred_digit = digit_logits.argmax(-1)
                    pred_full = full_logits.argmax(-1)
                    val_correct_rot += (rot_logits.argmax(-1) == yb_rot).sum().item()
                val_correct_digit += (pred_digit == yb_digit).sum().item()
                val_correct_full += (pred_full == yb_full).sum().item()
                val_seen += len(xb)
                val_digit_preds.append(pred_digit.cpu())
                val_digit_true.append(yb_digit.cpu())
                val_full_preds.append(pred_full.cpu())
                val_full_true.append(yb_full.cpu())
        print(f"  epoch {epoch+1}/{cnn_epochs}: train_loss={total_loss/n_seen:.4f} "
              f"val_digit_acc={(val_correct_digit/val_seen if val_seen else float('nan')):.3f} "
              f"val_rotation_acc={(val_correct_rot/val_seen if val_seen else float('nan')):.3f} "
              f"val_full_class_acc={(val_correct_full/val_seen if val_seen else float('nan')):.3f}")

    # ── Per-digit accuracy + confusion matrix on the FINAL epoch's val predictions ──
    # This shows which SPECIFIC digit pairs are the bottleneck (e.g. 1 vs 7,
    # 2 vs 5) rather than just one aggregate accuracy number that hides which
    # digits are actually the problem -- especially useful when moving to
    # harder, larger digit sets where overall accuracy alone doesn't tell you
    # where to focus (more/fewer epochs, different digit choices, etc.).
    if val_digit_preds:
        val_digit_preds = torch.cat(val_digit_preds).numpy()
        val_digit_true = torch.cat(val_digit_true).numpy()
        print(f"\nPer-digit validation accuracy (final epoch):")
        for di, digit in enumerate(digits):
            mask = val_digit_true == di
            if mask.sum() > 0:
                acc = (val_digit_preds[mask] == di).mean()
                print(f"  digit {digit}: {acc:.3f} ({mask.sum()} val samples)")

        try:
            from sklearn.metrics import confusion_matrix
            cm_digit = confusion_matrix(val_digit_true, val_digit_preds, labels=list(range(n_digits)))
            print(f"\nDigit confusion matrix (rows=true, cols=predicted, order={digits}):")
            header = "        " + "".join(f"{d:>6}" for d in digits)
            print(header)
            for di, digit in enumerate(digits):
                row = "".join(f"{cm_digit[di, dj]:>6}" for dj in range(n_digits))
                print(f"  {digit:>4}: {row}")
            print("  (off-diagonal entries show which digit pairs are confused with each other)\n")
        except Exception as e:
            print(f"  (confusion matrix skipped: {e})\n")

    # ── Full-class (every digit x rotation combination) accuracy + confusion ──
    # Only meaningful when full_class_weight > 0 (the head was actually trained).
    # This is the direct "how well does the network separate all N clusters"
    # metric you actually asked for, as opposed to the digit-only view above.
    if (full_class_weight > 0 or only_full_class_head) and val_full_preds:
        val_full_preds = torch.cat(val_full_preds).numpy()
        val_full_true = torch.cat(val_full_true).numpy()
        n_classes = n_digits * n_rots
        print(f"Per-cluster (full class) validation accuracy (final epoch):")
        for ci, name in enumerate(class_names):
            mask = val_full_true == ci
            if mask.sum() > 0:
                acc = (val_full_preds[mask] == ci).mean()
                print(f"  cluster {name}: {acc:.3f} ({mask.sum()} val samples)")

        try:
            from sklearn.metrics import confusion_matrix
            cm_full = confusion_matrix(val_full_true, val_full_preds, labels=list(range(n_classes)))
            print(f"\nFull-class confusion matrix (rows=true, cols=predicted, order={class_names}):")
            header = "          " + "".join(f"{n:>6}" for n in class_names)
            print(header)
            for ci, name in enumerate(class_names):
                row = "".join(f"{cm_full[ci, cj]:>6}" for cj in range(n_classes))
                print(f"  {name:>6}: {row}")
            print("  (off-diagonal entries show which SPECIFIC clusters -- digit+rotation "
                  "combos -- are confused with each other, e.g. would reveal a 6@180 <-> 9@0 "
                  "confusion directly if it exists)\n")
        except Exception as e:
            print(f"  (full-class confusion matrix skipped: {e})\n")

    model.eval()
    with torch.no_grad():
        codes = model.embed(all_images.to(device)).cpu().numpy().astype(np.float32)

    # ── Immediate silhouette report -- no separate visualize round trip needed ──
    try:
        from sklearn.metrics import silhouette_score
        print("Embedding quality check (on the full generated set, before pairing):")
        print(f"  digit silhouette:    {silhouette_score(codes, digit_idx):.3f}")
        print(f"  rotation silhouette: {silhouette_score(codes, rot_idx):.3f}")
        print(f"  full-class ({len(class_names)}-way) silhouette: {silhouette_score(codes, labels):.3f}")
        print("  (rough guide: <0.15 weak, 0.15-0.4 moderate, >0.4 strong separation)\n")
    except Exception as e:
        print(f"  (silhouette check skipped: {e})\n")

    out_perm = rng.permutation(len(codes))
    return codes[out_perm], labels[out_perm], class_names, digit_idx[out_perm], rot_idx[out_perm]


def generate_mnist_rotation_embeddings_unsupervised(
    digits,
    rotations,
    samples_per_class=200,
    embed_dim=10,
    seed=45,
    mnist_root="./mnist_data",
    ae_epochs=15,
    ae_lr=1e-3,
    ae_weight_decay=1e-5,
    ae_batch_size=64,
    val_fraction=0.15,
    l2_normalize_embed=True,
    aug_consistency_weight=0.0,
    device=None,
):
    """Genuinely unsupervised alternative to generate_mnist_rotation_embeddings().
    Trains a convolutional AUTOENCODER on the rotated images using ONLY a
    pixel reconstruction loss -- digit identity and rotation angle are
    NEVER used ANYWHERE in training. No labels of any kind, no pairwise
    term, nothing -- the dataset fed to the training loop contains only
    images (see the TensorDataset below, which is constructed from
    all_images alone). labels/digit_idx/rot_idx are computed and returned
    purely for POST-TRAINING evaluation (silhouette scores, the diagnostic
    KMeans-vs-ground-truth comparison below) -- never passed into the
    model, the optimizer, or the loss function during training.

    By design, pairwise/contrastive supervision does NOT live here --
    that belongs in DeepDPM's --contrastive_weight /
    --subcluster_contrastive_weight, operating on top of this embedding.
    Keeping the two separate means any effect the pairwise loss has during
    DeepDPM training is real, not confounded by the embedding already
    having been shaped by label information.

    Why this is the right tool for "should 6@180 and 9@0 end up in the
    same cluster": the supervised generator's digit/rotation/full-class
    heads are all trained with a loss that explicitly PUNISHES confusing
    any two classes -- by construction, they can never merge visually
    similar classes, no matter how alike the pixels actually look. This
    autoencoder has no such constraint: the embedding is shaped purely by
    what minimizes reconstruction error, so if two (digit, rotation)
    combinations are genuinely close in pixel space, nothing stops the
    encoder from placing them close together too. Whatever structure
    emerges here reflects real visual similarity, not enforced separation
    -- this may mean digit clusters are noisier/less separated than the
    supervised version, and that's expected, not a bug.

    aug_consistency_weight: pure reconstruction is a WEAK objective for
    clustering -- an autoencoder with enough capacity can just learn to
    copy pixels, with no pressure to organize similar inputs near each
    other. Setting this > 0 adds a self-supervised consistency term:
    each image gets two independent augmented views (small translation,
    brightness/contrast jitter, mild Gaussian noise -- deliberately NOT
    rotation, since that's the axis you want preserved as real structure,
    not collapsed via invariance), and the encoder is trained so both
    views land close together in embedding space (MSE between their
    embeddings). This still never touches any digit/rotation label -- it
    only uses the fact that two augmented copies came from the same
    source image. Empirically this tends to produce much more
    clustering-friendly embeddings than reconstruction alone, since it
    directly rewards grouping similar content rather than only pixel
    fidelity. 0 (default) = pure reconstruction, unchanged from before;
    try 0.5-1.0 if silhouette scores are weak.

    Returns the same 5-tuple as generate_mnist_rotation_embeddings(), so
    every downstream pairing function works unchanged on top of it.
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

    if isinstance(samples_per_class, (list, tuple)):
        if len(samples_per_class) != n_digits:
            raise ValueError(f"samples_per_class list has {len(samples_per_class)} entries "
                              f"but there are {n_digits} digits -- must match, or pass a single int.")
        samples_per_digit = list(samples_per_class)
    else:
        samples_per_digit = [samples_per_class] * n_digits
    print(f"Samples per digit: {dict(zip(digits, samples_per_digit))}")

    all_images = []
    labels = []
    digit_idx_list = []
    rot_idx_list = []
    class_names = []

    class_idx = 0
    for di, digit in enumerate(digits):
        n_samples_this_digit = samples_per_digit[di]
        pool_mask = dataset.targets == digit
        pool = dataset.data[pool_mask]
        if len(pool) < n_samples_this_digit:
            raise ValueError(f"Not enough MNIST images for digit {digit}: requested "
                              f"{n_samples_this_digit}, only {len(pool)} available.")
        chosen = rng.choice(len(pool), size=n_samples_this_digit, replace=False)
        base_imgs = pool[chosen].float()

        for ri, rot in enumerate(rotations):
            rotated = torch.stack([
                TF.rotate(img.unsqueeze(0), angle=float(rot)).squeeze(0)
                for img in base_imgs
            ])
            all_images.append((rotated / 255.0).unsqueeze(1))
            labels.append(np.full(n_samples_this_digit, class_idx, dtype=np.int64))
            digit_idx_list.append(np.full(n_samples_this_digit, di, dtype=np.int64))
            rot_idx_list.append(np.full(n_samples_this_digit, ri, dtype=np.int64))
            class_names.append(f"{digit}_{ri}")
            class_idx += 1

    all_images = torch.cat(all_images, dim=0)
    labels = np.concatenate(labels)
    digit_idx = np.concatenate(digit_idx_list)
    rot_idx = np.concatenate(rot_idx_list)

    n = len(all_images)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # NOTE: only images go into the dataset -- no labels of any kind. This
    # is what makes the training loop below genuinely unsupervised.
    train_ds = TensorDataset(all_images[train_idx])
    val_ds = TensorDataset(all_images[val_idx])
    train_dl = DataLoader(train_ds, batch_size=ae_batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=ae_batch_size, shuffle=False)

    class ConvAutoencoder(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            # encoder -- mirrors the supervised model's conv trunk
            self.enc_conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.enc_bn1 = nn.BatchNorm2d(16)
            self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.enc_bn2 = nn.BatchNorm2d(32)
            self.enc_conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.enc_bn3 = nn.BatchNorm2d(64)
            self.fc_embed = nn.Linear(64 * 7 * 7, embed_dim)
            self.embed_bn = nn.BatchNorm1d(embed_dim)

            # decoder -- mirrors the encoder in reverse
            self.fc_decode = nn.Linear(embed_dim, 64 * 7 * 7)
            self.dec_conv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 7->14
            self.dec_bn1 = nn.BatchNorm2d(32)
            self.dec_conv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 14->28
            self.dec_bn2 = nn.BatchNorm2d(16)
            self.dec_conv3 = nn.Conv2d(16, 1, 3, padding=1)

        def encode(self, x):
            x = Fnn.max_pool2d(Fnn.relu(self.enc_bn1(self.enc_conv1(x))), 2)  # 28->14
            x = Fnn.max_pool2d(Fnn.relu(self.enc_bn2(self.enc_conv2(x))), 2)  # 14->7
            x = Fnn.relu(self.enc_bn3(self.enc_conv3(x)))                    # 7->7
            x = x.flatten(1)                                                 # (B, 64*7*7)
            e = self.embed_bn(self.fc_embed(x))
            return Fnn.normalize(e, dim=1) if l2_normalize_embed else e

        def decode(self, e):
            x = self.fc_decode(e).view(-1, 64, 7, 7)
            x = Fnn.relu(self.dec_bn1(self.dec_conv1(x)))   # 7->14
            x = Fnn.relu(self.dec_bn2(self.dec_conv2(x)))   # 14->28
            x = torch.sigmoid(self.dec_conv3(x))            # pixel values in [0,1]
            return x

        def forward(self, x):
            e = self.encode(x)
            return self.decode(e), e

    def augment_batch(xb, rng_gen):
        """Small, benign augmentations for the self-supervised consistency
        term -- deliberately NOT rotation, since rotation is real structure
        we want the embedding to preserve, not collapse via invariance.
        Operates on a batch of (B, 1, 28, 28) tensors in [0, 1].
        """
        b = xb.shape[0]
        out = xb.clone()

        # small random translation (+/- 2 px), via roll + zero-out wrapped edges
        shift_x = torch.randint(-2, 3, (1,), generator=rng_gen).item()
        shift_y = torch.randint(-2, 3, (1,), generator=rng_gen).item()
        if shift_x != 0:
            out = torch.roll(out, shifts=shift_x, dims=3)
            if shift_x > 0:
                out[:, :, :, :shift_x] = 0
            else:
                out[:, :, :, shift_x:] = 0
        if shift_y != 0:
            out = torch.roll(out, shifts=shift_y, dims=2)
            if shift_y > 0:
                out[:, :, :shift_y, :] = 0
            else:
                out[:, :, shift_y:, :] = 0

        # random brightness/contrast jitter, per-sample
        brightness = 1.0 + (torch.rand(b, 1, 1, 1, generator=rng_gen) - 0.5) * 0.4  # [0.8, 1.2]
        contrast = 1.0 + (torch.rand(b, 1, 1, 1, generator=rng_gen) - 0.5) * 0.4
        mean = out.mean(dim=[2, 3], keepdim=True)
        out = (out - mean) * contrast.to(out.device) + mean
        out = out * brightness.to(out.device)

        # mild Gaussian noise
        out = out + torch.randn(out.shape, generator=rng_gen).to(out.device) * 0.05

        return out.clamp(0.0, 1.0)

    model = ConvAutoencoder(embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_lr, weight_decay=ae_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ae_epochs)
    aug_rng = torch.Generator().manual_seed(seed)

    print(f"Training convolutional autoencoder ({ae_epochs} epochs, embed_dim={embed_dim}, "
          f"l2_normalize={l2_normalize_embed}, aug_consistency_weight={aug_consistency_weight}) "
          f"-- NO labels of any kind used in training...")
    for epoch in range(ae_epochs):
        model.train()
        total_recon_loss, total_consistency_loss, n_seen = 0.0, 0.0, 0
        for (xb,) in train_dl:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, e = model(xb)
            loss = Fnn.mse_loss(recon, xb)

            if aug_consistency_weight > 0:
                view1 = augment_batch(xb.cpu(), aug_rng).to(device)
                view2 = augment_batch(xb.cpu(), aug_rng).to(device)
                _, e1 = model(view1)
                _, e2 = model(view2)
                consistency_loss = Fnn.mse_loss(e1, e2)
                loss = loss + aug_consistency_weight * consistency_loss
                total_consistency_loss += consistency_loss.item() * len(xb)

            loss.backward()
            optimizer.step()
            total_recon_loss += Fnn.mse_loss(recon, xb).item() * len(xb)
            n_seen += len(xb)
        scheduler.step()

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(device)
                recon, _ = model(xb)
                val_loss += Fnn.mse_loss(recon, xb).item() * len(xb)
                val_seen += len(xb)

        log_line = (f"  epoch {epoch+1}/{ae_epochs}: train_recon_loss={total_recon_loss/n_seen:.5f} "
                    f"val_recon_loss={(val_loss/val_seen if val_seen else float('nan')):.5f}")
        if aug_consistency_weight > 0:
            log_line += f" consistency_loss={total_consistency_loss/n_seen:.5f}"
        print(log_line)

    model.eval()
    with torch.no_grad():
        codes = model.encode(all_images.to(device)).cpu().numpy().astype(np.float32)

    # ── Diagnostic only (NOT used to shape the embedding): does unsupervised
    # KMeans on this embedding recover digit/rotation/full-class structure,
    # and specifically -- does it merge any visually-ambiguous combinations
    # (e.g. would a 6@180 and a 9@0 land in the same KMeans cluster)? ──
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, confusion_matrix
        n_classes = n_digits * n_rots

        print("\nEmbedding quality check (on the full generated set; NO labels were used in training):")
        print(f"  digit silhouette:    {silhouette_score(codes, digit_idx):.3f}")
        print(f"  rotation silhouette: {silhouette_score(codes, rot_idx):.3f}")
        print(f"  full-class ({n_classes}-way) silhouette: {silhouette_score(codes, labels):.3f}")
        print("  (rough guide: <0.15 weak, 0.15-0.4 moderate, >0.4 strong separation)")

        km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed)
        km_labels = km.fit_predict(codes)
        cm = confusion_matrix(labels, km_labels, labels=list(range(n_classes)))
        # for each true class, which KMeans cluster does it mostly land in
        print(f"\nUnsupervised KMeans (k={n_classes}) vs. ground-truth class -- "
              f"dominant KMeans cluster per true class (diagnostic only, not used in training):")
        for ci, name in enumerate(class_names):
            row = cm[ci]
            if row.sum() > 0:
                dominant = row.argmax()
                purity = row[dominant] / row.sum()
                # if two ground-truth classes share the same dominant KMeans
                # cluster, that's evidence the embedding genuinely merged them
                print(f"  true class {name}: {purity:.2f} of its points fall in KMeans cluster {dominant}")
        print("  (if two different true classes list the SAME dominant KMeans cluster, "
              "the unsupervised embedding is genuinely grouping them together -- e.g. "
              "this is where you'd see a real 6@180 <-> 9@0 merge if it exists in the pixels)\n")
    except Exception as e:
        print(f"  (embedding quality / KMeans diagnostic skipped: {e})\n")

    out_perm = rng.permutation(len(codes))
    return codes[out_perm], labels[out_perm], class_names, digit_idx[out_perm], rot_idx[out_perm]


# ---------------------------------------------------------------------------
# Step 4: build pairs from EXACTLY the given rotation-pair combinations
# ---------------------------------------------------------------------------
def build_custom_rotation_pairs(codes, digit_idx, rot_idx, labels, digits, rotations,
                                 rotation_pairs, pairs_per_combo=5, seed=45):
    """For each digit, and for each (ra, rb) in rotation_pairs (given as
    actual rotation VALUES, e.g. degrees), draws pairs_per_combo
    anchor/partner pairs with anchor rotation ra and partner rotation rb.
    z = 1.0 if ra == rb, else 0.0. Only the listed combinations are
    produced -- nothing else.

    pairs_per_combo: either a single int (broadcast -- same number of
    pairs drawn per combination for every digit) OR a list of ints, one
    per digit in `digits` (e.g. digits=[3,4,7], pairs_per_combo=[50,10,5]
    draws 50 pairs per rotation-combo for digit 3, 10 for digit 4, 5 for
    digit 7 -- lets you generate MORE pairwise supervision for classes you
    care more about, or where you specifically want to test whether more
    pair supervision helps a harder class).

    Returns:
        paired_codes: (M, 2, D) float32
        paired_labels: (M,) int64 -- anchor's full class id
        pair_labels: (M,) float32
        partner_labels: (M,) int64 -- partner's full class id (for the sample printout)
    """
    rng = np.random.default_rng(seed)
    n_digits = len(digits)
    rot_value_to_idx = {r: i for i, r in enumerate(rotations)}

    if isinstance(pairs_per_combo, (list, tuple)):
        if len(pairs_per_combo) != n_digits:
            raise ValueError(f"pairs_per_combo list has {len(pairs_per_combo)} entries "
                              f"but there are {n_digits} digits -- must match, or pass a single int.")
        pairs_per_combo_per_digit = list(pairs_per_combo)
    else:
        pairs_per_combo_per_digit = [pairs_per_combo] * n_digits
    print(f"Pairs per rotation-combo per digit: {dict(zip(digits, pairs_per_combo_per_digit))}")

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
        n_pairs_this_digit = pairs_per_combo_per_digit[di]
        for ra_val, rb_val in rotation_pairs:
            ra, rb = rot_value_to_idx[ra_val], rot_value_to_idx[rb_val]
            anchor_pool = by_digit_rot[(di, ra)]
            partner_pool = by_digit_rot[(di, rb)]
            z = 1.0 if ra == rb else 0.0

            for _ in range(n_pairs_this_digit):
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


def build_cross_digit_pairs(codes, digit_idx, labels, digits, digit_pairs, pairs_per_combo=50, seed=45,
                             hard_negative_mining=False):
    """Builds NEGATIVE (z=0) pairs between specific pairs of DIGITS,
    ignoring rotation entirely (anchor and partner can be at any rotation).

    This exists specifically to give DeepDPM's --contrastive_weight a
    signal for digit-level confusion that an UNSUPERVISED embedding can
    have but a supervised one cannot (supervised heads eliminate any
    cross-digit confusion by construction, so cross-digit pairs would be
    redundant there -- but for an unsupervised autoencoder, two digits
    that are genuinely visually similar (e.g. 3 and 5) may end up close
    or overlapping in the embedding, and nothing in reconstruction loss
    alone tells the encoder they should be different. This function
    builds the training signal to test whether pairwise supervision in
    DeepDPM can pull them apart afterward.

    Args:
        digit_pairs: list of (digit_a, digit_b) tuples using actual digit
            values (not indices), e.g. [(3, 5)] -- the specific digit
            pairs you want negative supervision for.
        pairs_per_combo: how many (anchor, partner) pairs to draw per
            digit_pair.
        hard_negative_mining: if False (default), anchor/partner are drawn
            UNIFORMLY AT RANDOM from the two digit pools -- most such
            pairs are already easy (far apart in embedding space) and
            contribute little signal. If True, instead finds the
            pairs_per_combo pairs with the SMALLEST distance in embedding
            space across the two pools -- i.e. the specific examples the
            feature extractor is actually confusing right now. This
            concentrates supervision exactly where it's needed instead of
            spreading it randomly, and still only uses the digit label you
            already have (no new information source, just used more
            efficiently). Requires computing a distance matrix between the
            two digit pools (O(n_a * n_b * embed_dim)) -- fine at typical
            per-digit sample sizes (hundreds to low thousands), but can get
            slow at very large pools.

    Returns the same 4-tuple shape as build_custom_rotation_pairs, so it
    can be concatenated with rotation pairs before saving.
    """
    rng = np.random.default_rng(seed)
    digit_value_to_idx = {d: i for i, d in enumerate(digits)}

    for da, db in digit_pairs:
        if da not in digit_value_to_idx or db not in digit_value_to_idx:
            raise ValueError(f"digit_pairs entry ({da}, {db}) uses a value not in --digits {digits}.")
        if da == db:
            raise ValueError(f"digit_pairs entry ({da}, {db}) has the same digit twice -- "
                              f"cross-digit pairs must be between two DIFFERENT digits. "
                              f"Use --rotation-pairs for within-digit supervision instead.")

    by_digit = {}
    for di in range(len(digits)):
        by_digit[di] = np.where(digit_idx == di)[0]
        if len(by_digit[di]) == 0:
            raise ValueError(f"No points found for digit index {di}.")

    paired_codes, paired_labels, pair_labels, partner_labels = [], [], [], []

    for da_val, db_val in digit_pairs:
        da, db = digit_value_to_idx[da_val], digit_value_to_idx[db_val]
        anchor_pool = by_digit[da]
        partner_pool = by_digit[db]

        if hard_negative_mining:
            # find the pairs_per_combo GLOBALLY closest (anchor, partner)
            # combinations across the two pools -- the actual confusions,
            # not random draws.
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(codes[anchor_pool], codes[partner_pool])
            k = min(pairs_per_combo, dist_matrix.size)
            flat_idx = np.argpartition(dist_matrix, k - 1, axis=None)[:k]
            rows, cols = np.unravel_index(flat_idx, dist_matrix.shape)
            mean_dist = dist_matrix[rows, cols].mean()
            print(f"  hard-negative mining ({digits[da]} vs {digits[db]}): "
                  f"selected {k} closest pairs, mean embedding distance={mean_dist:.4f} "
                  f"(vs. overall pool mean distance={dist_matrix.mean():.4f})")
            selected_anchor = anchor_pool[rows]
            selected_partner = partner_pool[cols]
            for a_idx, p_idx in zip(selected_anchor, selected_partner):
                paired_codes.append(np.stack([codes[a_idx], codes[p_idx]], axis=0))
                paired_labels.append(labels[a_idx])
                pair_labels.append(0.0)
                partner_labels.append(labels[p_idx])
        else:
            for _ in range(pairs_per_combo):
                anchor_idx = int(rng.choice(anchor_pool))
                partner_idx = int(rng.choice(partner_pool))
                paired_codes.append(np.stack([codes[anchor_idx], codes[partner_idx]], axis=0))
                paired_labels.append(labels[anchor_idx])
                pair_labels.append(0.0)   # always a negative pair -- different digits
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
    parser.add_argument("--digit-pairs", type=str, default=None,
                         help="Comma-separated 'a-b' DIGIT pairs (e.g. '3-5') to generate cross-digit NEGATIVE "
                              "pairs for, ignoring rotation. Useful specifically for unsupervised embeddings, "
                              "where two visually similar digits (e.g. 3 and 5) may not be separated at all -- "
                              "supervised embeddings never need this since their heads already eliminate any "
                              "cross-digit confusion by construction. These pairs are merged into the same "
                              "paired dataset as --rotation-pairs, giving DeepDPM's --contrastive_weight a "
                              "digit-separation signal alongside the rotation one. Omit to skip (default).")
    parser.add_argument("--digit-pairs-per-combo", type=int, default=50,
                         help="How many cross-digit pairs to draw per --digit-pairs combination. Default 50.")
    parser.add_argument("--hard-negative-mining", action="store_true",
                         help="For --digit-pairs: instead of random anchor/partner draws, select the pairs "
                              "that are CLOSEST together in embedding space -- i.e. the specific examples the "
                              "feature extractor is actually confusing, rather than random (mostly already-easy) "
                              "pairs. Concentrates pairwise supervision exactly where it's needed.")
    parser.add_argument("--data-source", type=str, default="supervised", choices=["supervised", "unsupervised"],
                         help="'supervised' (default): trains digit + rotation + full-class heads -- guarantees "
                              "cluster separation by construction, cannot merge visually-similar classes. "
                              "'unsupervised': trains a convolutional AUTOENCODER with reconstruction loss only, "
                              "no labels used in training at all -- whatever structure emerges reflects real "
                              "pixel-level visual similarity, so genuinely ambiguous combinations (e.g. a rotated "
                              "6 that looks like a 9) CAN end up close together or even merged, unlike the "
                              "supervised path. Use this to test whether visual ambiguity is real vs. imposed.")
    parser.add_argument("--samples-per-class", type=int, nargs="+", default=[1000],
                         help="Images per digit (each expanded across every rotation). Pass a single value to use "
                              "the same count for every digit (e.g. --samples-per-class 500), or one value per "
                              "digit in the SAME ORDER as --digits (e.g. --digits 3 4 7 --samples-per-class 500 800 300 "
                              "pulls 500 of digit 3, 800 of digit 4, 300 of digit 7).")
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--pairs-per-combo", type=int, nargs="+", default=[5],
                         help="Pairs drawn per rotation-combo, per digit. Pass a single value to use the same "
                              "count for every digit (e.g. --pairs-per-combo 20), or one value per digit in the "
                              "SAME ORDER as --digits (e.g. --digits 3 4 7 --pairs-per-combo 50 10 5 draws 50 pairs "
                              "per combo for digit 3, 10 for digit 4, 5 for digit 7).")
    parser.add_argument("--cnn-epochs", type=int, default=15, help="[supervised only]")
    parser.add_argument("--cnn-lr", type=float, default=1e-3, help="[supervised only]")
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4, help="[supervised only]")
    parser.add_argument("--cnn-batch-size", type=int, default=64, help="[supervised only]")
    parser.add_argument("--cnn-dropout", type=float, default=0.3, help="[supervised only]")
    parser.add_argument("--cnn-val-fraction", type=float, default=0.15, help="[supervised only]")
    parser.add_argument("--no-l2-normalize", action="store_true",
                         help="Disable L2-normalizing the final embedding (normalized by default -- "
                              "keeps distances meaningful for downstream clustering).")
    parser.add_argument("--rotation-aux-weight", type=float, default=0.3, help="[supervised only]. "
                         "Weight on the auxiliary rotation-prediction head. 0 = pure digit classifier "
                         "(becomes rotation-invariant, erases rotation structure). Default 0.3.")
    parser.add_argument("--full-class-weight", type=float, default=1.0, help="[supervised only]. "
                         "Weight on a THIRD head trained directly on the full joint (digit, rotation) class -- "
                         "i.e. each cluster as its own distinct target, not just digit/rotation separately. "
                         "Default 1.0 (on). Set to 0 to train with only the digit + rotation marginal heads.")
    parser.add_argument("--only-full-class-head", action="store_true", help="[supervised only]. "
                         "Removes the separate digit and rotation heads entirely -- trains ONLY the combined "
                         "full-class head (one label per (digit,rotation) combination). --rotation-aux-weight "
                         "and --full-class-weight are ignored in this mode. Digit/rotation accuracy are still "
                         "reported, derived post-hoc by decoding the predicted full class.")
    parser.add_argument("--ae-epochs", type=int, default=15, help="[unsupervised only] autoencoder training epochs")
    parser.add_argument("--ae-lr", type=float, default=1e-3, help="[unsupervised only]")
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5, help="[unsupervised only]")
    parser.add_argument("--ae-batch-size", type=int, default=64, help="[unsupervised only]")
    parser.add_argument("--ae-val-fraction", type=float, default=0.15, help="[unsupervised only]")
    parser.add_argument("--aug-consistency-weight", type=float, default=0.0, help="[unsupervised only]. "
                         "Self-supervised consistency term: encourages two independently-augmented views "
                         "(small translation, brightness/contrast jitter, noise -- NOT rotation) of the same "
                         "image to land close together in embedding space. Never uses digit/rotation labels. "
                         "0 (default) = pure reconstruction. Try 0.5-1.0 if silhouette scores are weak.")
    parser.add_argument("--mnist-root", type=str, default="./mnist_data")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    # resolve --samples-per-class: either a single value (broadcast) or
    # one per digit, in the same order as --digits
    if len(args.samples_per_class) == 1:
        samples_per_class = args.samples_per_class[0]
    elif len(args.samples_per_class) == len(args.digits):
        samples_per_class = args.samples_per_class
    else:
        print(f"ERROR: --samples-per-class got {len(args.samples_per_class)} value(s) "
              f"but --digits has {len(args.digits)} entries. Pass either a single value "
              f"(same count for every digit) or exactly one value per digit.")
        sys.exit(1)

    # resolve --pairs-per-combo: either a single value (broadcast) or
    # one per digit, in the same order as --digits
    if len(args.pairs_per_combo) == 1:
        pairs_per_combo = args.pairs_per_combo[0]
    elif len(args.pairs_per_combo) == len(args.digits):
        pairs_per_combo = args.pairs_per_combo
    else:
        print(f"ERROR: --pairs-per-combo got {len(args.pairs_per_combo)} value(s) "
              f"but --digits has {len(args.digits)} entries. Pass either a single value "
              f"(same count for every digit) or exactly one value per digit.")
        sys.exit(1)

    # parse rotation-pairs and infer the rotation set from it
    rotation_pairs = []
    for token in args.rotation_pairs.split(","):
        a, b = token.strip().split("-")
        rotation_pairs.append((int(a), int(b)))
    rotations = sorted(set(r for pair in rotation_pairs for r in pair))
    print(f"Inferred rotations from --rotation-pairs: {rotations}")

    if args.data_source == "unsupervised":
        codes, labels, class_names, digit_idx, rot_idx = generate_mnist_rotation_embeddings_unsupervised(
            digits=args.digits,
            rotations=rotations,
            samples_per_class=samples_per_class,
            embed_dim=args.embed_dim,
            seed=args.seed,
            mnist_root=args.mnist_root,
            ae_epochs=args.ae_epochs,
            ae_lr=args.ae_lr,
            ae_weight_decay=args.ae_weight_decay,
            ae_batch_size=args.ae_batch_size,
            val_fraction=args.ae_val_fraction,
            aug_consistency_weight=args.aug_consistency_weight,
            l2_normalize_embed=not args.no_l2_normalize,
        )
    else:
        codes, labels, class_names, digit_idx, rot_idx = generate_mnist_rotation_embeddings(
            digits=args.digits,
            rotations=rotations,
            samples_per_class=samples_per_class,
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
            full_class_weight=args.full_class_weight,
            only_full_class_head=args.only_full_class_head,
            l2_normalize_embed=not args.no_l2_normalize,
        )

    # full unpaired export first -- always, regardless of pairing subsetting
    save_unpaired_dataset(args.out_dir, codes, labels, split="train")

    paired_codes, paired_labels, pair_labels, partner_labels = build_custom_rotation_pairs(
        codes, digit_idx, rot_idx, labels,
        digits=args.digits, rotations=rotations,
        rotation_pairs=rotation_pairs,
        pairs_per_combo=pairs_per_combo,
        seed=args.seed,
    )

    parsed_digit_pairs = None
    if args.digit_pairs:
        parsed_digit_pairs = []
        for token in args.digit_pairs.split(","):
            a, b = token.strip().split("-")
            parsed_digit_pairs.append((int(a), int(b)))

        cd_codes, cd_labels, cd_pair_labels, cd_partner_labels = build_cross_digit_pairs(
            codes, digit_idx, labels,
            digits=args.digits,
            digit_pairs=parsed_digit_pairs,
            pairs_per_combo=args.digit_pairs_per_combo,
            seed=args.seed + 2,   # different seed stream than rotation pairs
            hard_negative_mining=args.hard_negative_mining,
        )
        print(f"Adding {len(cd_pair_labels)} cross-digit negative pairs for {parsed_digit_pairs} "
              f"to the paired dataset (rotation ignored for these).")

        paired_codes = np.concatenate([paired_codes, cd_codes], axis=0)
        paired_labels = np.concatenate([paired_labels, cd_labels], axis=0)
        pair_labels = np.concatenate([pair_labels, cd_pair_labels], axis=0)
        partner_labels = np.concatenate([partner_labels, cd_partner_labels], axis=0)

        # reshuffle the combined set so rotation-pairs and cross-digit-pairs are interleaved
        merge_rng = np.random.default_rng(args.seed + 3)
        merge_perm = merge_rng.permutation(len(paired_codes))
        paired_codes = paired_codes[merge_perm]
        paired_labels = paired_labels[merge_perm]
        pair_labels = pair_labels[merge_perm]
        partner_labels = partner_labels[merge_perm]

    save_paired_dataset(args.out_dir, paired_codes, paired_labels, pair_labels, split="train")

    save_metadata(
        args.out_dir, class_names,
        meta_extra={
            "digits": args.digits,
            "rotations": rotations,
            "rotation_pairs": rotation_pairs,
            "digit_pairs": parsed_digit_pairs,
            "samples_per_class": samples_per_class,
            "embed_dim": args.embed_dim,
            "pairs_per_combo": pairs_per_combo,
            "cnn_epochs": args.cnn_epochs,
            "data_source": args.data_source,
            "rotation_aux_weight": args.rotation_aux_weight,
            "full_class_weight": args.full_class_weight,
            "only_full_class_head": args.only_full_class_head,
            "seed": args.seed,
        },
    )

    print_pair_sample(paired_labels, pair_labels, partner_labels, class_names, n_samples=20, seed=args.seed)


if __name__ == "__main__":
    main()