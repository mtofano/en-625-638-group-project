#!/usr/bin/env python3
"""
ResNet-18 with minority-focused sampling, class-conditional augmentation, early stopping,
and **epoch-wise adaptive re-sampling** driven by a confusion matrix (default: from validation).

- No argparse: everything is controlled by constants below.
- Zero leakage: split per-record_id within each species folder.
- No normalization (match your baseline); images assumed 224x224 RGB.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import random, io, csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet18

# =========================
# Constants (edit here)
# =========================
IMAGES_ROOT        = Path("foundations_neural_networks/PetFace/images")
OUT_DIR            = Path("foundations_neural_networks/PetFace/runs/cnn_folder_224_raw_aug")

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
SEED               = 42
BATCH_SIZE         = 128
EPOCHS             = 30
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4
NUM_WORKERS        = 4
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP            = True
CHECKPOINT_EVERY   = 1
RESUME             = True
PRINT_EVERY        = 50

# ---- Minority definition (choose ONE mode via TAIL_MODE) ----
TAIL_MODE              = "head_exclusion"   # options: "quantile", "absolute", "head_exclusion"
TAIL_Q                 = 0.85               # if quantile: bottom 85% by count are minority
ABSOLUTE_TAIL_MAX      = 2500               # if absolute: count <= this is minority
HEAD_EXCLUDE           = {"cat", "dog"}     # if head_exclusion: these are never minority
HEAD_EXCLUSION_MAX     = 80_000            # and others with count <= this are minority

# ---- Weighted sampling (base) ----
MAX_UPWEIGHT_PER_CLASS = 3.0         # cap on any class's sampling weight
SAMPLING_ALPHA         = 1.0         # weight_c ~ (median_count / count_c)^alpha, then clipped in [1, MAX_UPWEIGHT]

# ---- Augmentation knobs ----
ERASING_BASE_P         = 0.25
ERASING_TAIL_P         = 0.40

# ---- Early stopping ----
EARLY_STOP_PATIENCE    = 7
EARLY_STOP_DELTA       = 1e-4

# ---- Optional confusion CSV (from a previous run) ----
# (Not used for adaptation-in-epoch; kept for logging parity with your previous setup)
CONFUSION_CSV          = None
USE_CONFUSION_FOR_SAMPLING = False   # If True and CONFUSION_CSV exists, only affects initial epoch weights.

ALLOW_EXTS = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}

# ---- NEW: Dynamic adaptation knobs ----
ADAPTIVE_SAMPLING      = True        # If True, update sampler weights *after every epoch* using confusion-derived recall.
ADAPT_ON_SPLIT         = "val"       # "val" or "train" — which split to build the confusion from.
ADAPT_START_EPOCH      = 0           # First epoch after which to adapt (0 = adapt immediately after epoch 0).
PERF_ALPHA             = 1.0         # How strongly poor recall boosts sampling: multiplier ~ (difficulty/median_difficulty)^PERF_ALPHA
MIN_EVAL_SUPPORT       = 20          # If a class has < this many true examples in the evaluated split, treat recall as "unknown" (smoothed).
DIFFICULTY_FLOOR       = 0.05        # Prevent division by near-zero when many classes are easy; floor for median difficulty.
SAVE_CONFUSIONS        = True        # Save a CSV confusion matrix each epoch (in OUT_DIR/confusions/)
# =========================

# Seeding
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONF_DIR = OUT_DIR / "confusions"
if SAVE_CONFUSIONS:
    CONF_DIR.mkdir(parents=True, exist_ok=True)

# --------- helpers: splitting with zero leakage ---------
def compute_sizes(n: int, fracs: Tuple[float,float,float]) -> Tuple[int,int,int]:
    t, v, r = fracs
    s = t+v+r
    t, v, r = t/s, v/s, r/s
    ti, vi, ri = int(n*t), int(n*v), int(n*r)
    rem = n - (ti+vi+ri)
    ti += rem  # deterministic: remainder to train
    return ti, vi, ri

def list_species(root: Path) -> List[str]:
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def list_records(species_dir: Path) -> List[Path]:
    return sorted([p for p in species_dir.iterdir() if p.is_dir()])

def list_images(rec_dir: Path) -> List[Path]:
    return [p for p in rec_dir.rglob("*") if p.is_file() and p.suffix.lower() in ALLOW_EXTS]

# --------- build lists (paths + labels) ---------
@dataclass
class SplitLists:
    train: List[Tuple[Path,int]]
    val:   List[Tuple[Path,int]]
    test:  List[Tuple[Path,int]]

def build_split_lists(images_root: Path) -> Tuple[SplitLists, Dict[str,int]]:
    species = list_species(images_root)
    class_to_index = {sp: i for i, sp in enumerate(species)}

    train_items: List[Tuple[Path,int]] = []
    val_items:   List[Tuple[Path,int]] = []
    test_items:  List[Tuple[Path,int]] = []

    rng = random.Random(SEED)

    for sp in species:
        sp_dir = images_root / sp
        rec_dirs = list_records(sp_dir)
        rng.shuffle(rec_dirs)  # deterministic per species

        n = len(rec_dirs)
        n_tr, n_v, n_te = compute_sizes(n, (TRAIN_FRAC, VAL_FRAC, TEST_FRAC))

        rec_train = rec_dirs[:n_tr]
        rec_val   = rec_dirs[n_tr:n_tr+n_v]
        rec_test  = rec_dirs[n_tr+n_v:n_tr+n_v+n_te]

        cid = class_to_index[sp]
        for rd in rec_train:
            imgs = list_images(rd)
            train_items.extend([(p, cid) for p in imgs])
        for rd in rec_val:
            imgs = list_images(rd)
            val_items.extend([(p, cid) for p in imgs])
        for rd in rec_test:
            imgs = list_images(rd)
            test_items.extend([(p, cid) for p in imgs])

        print(f"[{sp}] records total={n} -> train={n_tr}, val={n_v}, test={n_te}")

    return SplitLists(train_items, val_items, test_items), class_to_index

# --------- transforms (class-conditional) ---------
def has_randaugment() -> bool:
    return hasattr(T, "RandAugment")

def build_base_transform() -> T.Compose:
    ops = [
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5, fill=0),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.RandomErasing(p=ERASING_BASE_P, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0.0),
    ]
    return T.Compose(ops)

def build_tail_transform() -> T.Compose:
    ops: List[torch.nn.Module] = [
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.12), shear=7, fill=0),
    ]
    if has_randaugment():
        ops.insert(0, T.RandAugment(num_ops=2, magnitude=12))
    ops += [
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.RandomErasing(p=ERASING_TAIL_P, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ]
    return T.Compose(ops)

# --------- dataset ---------
class ListImageDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int]],
                 base_tf: Optional[T.Compose] = None,
                 tail_tf: Optional[T.Compose] = None,
                 tail_labels: Optional[Set[int]] = None):
        self.items = items
        self.base_tf = base_tf or T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
        self.tail_tf = tail_tf or self.base_tf
        self.tail_labels = tail_labels or set()

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with open(path, "rb") as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        x = self.tail_tf(img) if label in self.tail_labels else self.base_tf(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# --------- model ---------
def build_model(num_classes: int) -> nn.Module:
    m = resnet18(weights=None)              # from scratch to match your baseline
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# --------- counting / CSV helpers ---------
def counts_from_items(items: List[Tuple[Path,int]], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for _, c in items:
        counts[c] += 1
    return counts

def read_confusion_counts(path: Optional[Path]) -> Optional[Dict[str,int]]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        header = rows[0]
        class_names = header[1:]
        counts: Dict[str,int] = {name: 0 for name in class_names}
        for r in rows[1:]:
            row_name = r[0]
            vals = r[1:1+len(class_names)]
            s = 0
            for v in vals:
                try:
                    s += int(float(v))
                except:
                    s += 0
            counts[row_name] = s
        return counts
    except Exception as e:
        print(f"[warn] failed to parse confusion CSV: {e}")
        return None

# --------- minority selection ---------
def median_val(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0: return 0.0
    if n % 2 == 1:
        return float(s[n//2])
    else:
        return 0.5 * (s[n//2 - 1] + s[n//2])

def compute_tail_set(counts: List[int],
                     class_to_index: Dict[str,int],
                     index_to_class: Dict[int,str]) -> Set[int]:
    n = len(counts)
    order = sorted(range(n), key=lambda i: counts[i])  # ascending by count
    if TAIL_MODE == "quantile":
        k = int(max(1, round(TAIL_Q * n)))
        tail = set(order[:k])
    elif TAIL_MODE == "absolute":
        tail = {i for i,c in enumerate(counts) if c <= ABSOLUTE_TAIL_MAX}
    elif TAIL_MODE == "head_exclusion":
        head_ids = {i for i, name in index_to_class.items() if name in HEAD_EXCLUDE}
        tail = {i for i,c in enumerate(counts) if (i not in head_ids) and (c <= HEAD_EXCLUSION_MAX)}
    else:
        raise ValueError(f"Unknown TAIL_MODE={TAIL_MODE}")
    return {i for i in tail if counts[i] > 0}

# --------- sampling weights ---------
def build_count_based_class_weights(per_class_counts: List[int]) -> List[float]:
    med = median_val([float(x) for x in per_class_counts if x > 0]) or 1.0
    class_w: List[float] = []
    for cnt in per_class_counts:
        if cnt <= 0:
            w = MAX_UPWEIGHT_PER_CLASS
        else:
            base = (med / float(cnt)) ** SAMPLING_ALPHA
            w = min(MAX_UPWEIGHT_PER_CLASS, max(1.0, base))
        class_w.append(w)
    return class_w

def apply_performance_boost(class_weights: List[float],
                            recalls: List[Optional[float]]) -> List[float]:
    """Multiply class_weights by a difficulty-based factor derived from recall."""
    difficulties: List[float] = []
    for r in recalls:
        if r is None:
            difficulties.append(None)
        else:
            difficulties.append(max(0.0, 1.0 - r))
    # Use median of defined difficulties; floor to avoid near-zero explosions.
    valid = [d for d in difficulties if d is not None]
    med_d = median_val(valid) if valid else 0.0
    med_d = max(DIFFICULTY_FLOOR, med_d)

    boosted: List[float] = []
    for w, d in zip(class_weights, difficulties):
        if d is None:
            mult = 1.0  # unknown recall -> neutral
        else:
            mult = (d / med_d) ** PERF_ALPHA
        boosted.append(min(MAX_UPWEIGHT_PER_CLASS, max(1.0, w * mult)))
    return boosted

def class_weights_to_sample_weights(labels: List[int], class_weights: List[float]) -> List[float]:
    return [class_weights[y] for y in labels]

# --------- confusion & recall utilities ---------
@torch.no_grad()
def compute_confusion_and_recall(model: nn.Module,
                                 loader: DataLoader,
                                 num_classes: int,
                                 device: str) -> Tuple[List[List[int]], List[Optional[float]]]:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP and device.startswith("cuda")):
            logits = model(xb)
        preds = logits.argmax(dim=1)
        # accumulate confusion
        for t, p in zip(yb.view(-1), preds.view(-1)):
            conf[t.long(), p.long()] += 1

    recalls: List[Optional[float]] = []
    for c in range(num_classes):
        support = int(conf[c].sum().item())
        correct  = int(conf[c, c].item())
        if support >= MIN_EVAL_SUPPORT:
            recalls.append(correct / float(support) if support > 0 else None)
        else:
            recalls.append(None)  # not enough support -> neutral
    return conf.tolist(), recalls

def save_confusion_csv(conf_mat: List[List[int]], class_names: List[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(conf_mat):
            w.writerow([class_names[i]] + [int(x) for x in row])

# --------- train / eval / early stop ---------
def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch: int):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    step = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP and DEVICE.startswith("cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * xb.size(0)
        running_correct += (preds == yb).sum().item()
        running_count += xb.size(0)
        step += 1
        if step % PRINT_EVERY == 0:
            print(f"epoch {epoch} step {step}  "
                  f"loss={running_loss/max(1,running_count):.4f}  "
                  f"acc={running_correct/max(1,running_count):.4f}")
    return running_loss / max(1, running_count), running_correct / max(1, running_count)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = 0.0
    tot_correct = 0
    tot_count = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP and DEVICE.startswith("cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)
        tot_loss += loss.item() * xb.size(0)
        tot_correct += (preds == yb).sum().item()
        tot_count += xb.size(0)
    return tot_loss / max(1, tot_count), tot_correct / max(1, tot_count)

class EarlyStopper:
    def __init__(self, mode: str = "max", patience: int = 7, delta: float = 1e-4):
        assert mode in ("max", "min")
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.best = -float("inf") if mode == "max" else float("inf")
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        improved = (metric > self.best + self.delta) if self.mode == "max" else (metric < self.best - self.delta)
        if improved:
            self.best = metric
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.patience

# --------- main ---------
def main():
    # Build split lists (zero leakage)
    splits, class_to_index = build_split_lists(IMAGES_ROOT)
    num_classes = len(class_to_index)
    index_to_class = {v:k for k,v in class_to_index.items()}
    class_names = [index_to_class[i] for i in range(num_classes)]

    # Train-set counts (source of truth for minority selection)
    train_counts = counts_from_items(splits.train, num_classes)

    # Optional: read confusion CSV for logging / optional initial sampling
    conf_name_to_count = read_confusion_counts(CONFUSION_CSV) if CONFUSION_CSV else None
    if conf_name_to_count is not None:
        aligned = [conf_name_to_count.get(index_to_class[i], 0) for i in range(num_classes)]
        print("[info] Confusion CSV counts (aligned):")
        for i, c in enumerate(aligned):
            print(f"  {index_to_class[i]}: {c}")

    # Decide minority classes (based on TRAIN counts)
    tail_labels = compute_tail_set(train_counts, class_to_index, index_to_class)
    print(f"[minority] {len(tail_labels)} classes marked minority via {TAIL_MODE}.")
    for i in sorted(tail_labels):
        print(f"  - {index_to_class[i]}: train_count={train_counts[i]}")

    # Datasets with class-conditional transforms
    base_tf = build_base_transform()
    tail_tf = build_tail_transform()
    train_ds = ListImageDataset(splits.train, base_tf=base_tf, tail_tf=tail_tf, tail_labels=tail_labels)
    val_ds   = ListImageDataset(splits.val,   base_tf=base_tf)

    # Initial weights (count-based, optionally from prior confusion CSV)
    sampling_counts = None
    if USE_CONFUSION_FOR_SAMPLING and conf_name_to_count is not None:
        sampling_counts = [conf_name_to_count.get(index_to_class[i], train_counts[i]) for i in range(num_classes)]
    else:
        sampling_counts = train_counts

    base_class_weights = build_count_based_class_weights(sampling_counts)
    train_labels = [lbl for _, lbl in splits.train]
    sample_weights = class_weights_to_sample_weights(train_labels, base_class_weights)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS>0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS>0)

    # Model/optim/scaler
    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device="cuda", enabled=USE_AMP and DEVICE.startswith("cuda"))

    # Resume
    ckpt_path = OUT_DIR / "last.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if RESUME and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] loaded checkpoint at epoch {ckpt['epoch']}")

    # Early stopping
    early_stop = EarlyStopper(mode="max", patience=EARLY_STOP_PATIENCE, delta=EARLY_STOP_DELTA)

    best_val_acc = -1.0
    for epoch in range(start_epoch, EPOCHS):
        # ---- Train ----
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, epoch)

        # ---- Eval (val) ----
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        print(f"[epoch {epoch}] train: loss={tr_loss:.4f} acc={tr_acc:.4f} | val: loss={va_loss:.4f} acc={va_acc:.4f}")

        # ---- Build confusion and adapt for next epoch ----
        if ADAPTIVE_SAMPLING and (epoch >= ADAPT_START_EPOCH):
            if ADAPT_ON_SPLIT == "val":
                conf_mat, recalls = compute_confusion_and_recall(model, val_loader, num_classes, DEVICE)
                split_name = "val"
            else:
                # Evaluate on train (using CURRENT train_loader's transforms)
                # If you want a "clean" train confusion, make a DataLoader with base_tf only.
                conf_mat, recalls = compute_confusion_and_recall(model, train_loader, num_classes, DEVICE)
                split_name = "train"

            if SAVE_CONFUSIONS:
                save_confusion_csv(conf_mat, class_names, CONF_DIR / f"confusion_{split_name}_epoch{epoch:03d}.csv")

            # Count-based weights (fixed) -> boost by performance difficulty
            # We use **train_counts** as base; adaptation uses **recalls** from chosen split.
            base_class_weights = build_count_based_class_weights(train_counts)
            boosted_class_weights = apply_performance_boost(base_class_weights, recalls)
            new_sample_weights = class_weights_to_sample_weights(train_labels, boosted_class_weights)

            # Update sampler in-place (no new DataLoader constructed)
            sampler.weights = torch.as_tensor(new_sample_weights, dtype=torch.double)

            # Optional: small log of the biggest boosts
            report = sorted(
                [(index_to_class[i], base_class_weights[i], boosted_class_weights[i], recalls[i] if recalls[i] is not None else -1.0)
                 for i in range(num_classes)],
                key=lambda t: t[2], reverse=True
            )[:5]
            print("[adapt] top-5 class weights after boost (class, base_w, boosted_w, recall):")
            for name, bw, aw, rc in report:
                rc_str = f"{rc:.3f}" if rc >= 0 else "n/a"
                print(f"        {name:>12s}  base={bw:.2f}  boosted={aw:.2f}  recall={rc_str}")

        # ---- Checkpoints ----
        if (epoch % CHECKPOINT_EVERY == 0) or (epoch == EPOCHS - 1):
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "class_to_index": class_to_index,
            }, ckpt_path)
            print(f"[ckpt] saved -> {ckpt_path}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "class_to_index": class_to_index},
                       OUT_DIR / "best.pt")
            print(f"[best] val acc improved to {best_val_acc:.4f}")

        # ---- Early stop ----
        if early_stop.step(va_acc):
            print(f"[early stop] no val acc improvement for >{EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch}.")
            break

    print("Done.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
