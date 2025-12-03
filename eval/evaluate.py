#!/usr/bin/env python3
# Evaluate best model on train/val/test (zero-leakage splits) + ALL images.
# Produces overall accuracy, per-class precision/recall/F1, and confusion matrices (CSV + PNG).

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import io
import random
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================
# Constants (edit here)
# =========================
IMAGES_ROOT        = Path("foundations_neural_networks/PetFace/images")

# Match your dynamic trainer's OUT_DIR so we load the right checkpoints/artifacts
# (see train_resnet_minority_aug_dynamic.py)
OUT_DIR            = Path("foundations_neural_networks/PetFace/runs/cnn_folder_224_raw_aug_norm")

# These must match training so the split is identical (record_id-level, per species)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
SEED               = 42

BATCH_SIZE         = 256
NUM_WORKERS        = 4
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

ALLOW_EXTS = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
# =========================

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------- helpers: splitting with zero leakage ---------
def compute_sizes(n: int, fracs: Tuple[float,float,float]) -> Tuple[int,int,int]:
    t, v, r = fracs
    s = t+v+r
    t, v, r = t/s, v/s, r/s
    ti, vi, ri = int(n*t), int(n*v), int(n*r)
    rem = n - (ti+vi+ri)
    ti += rem  # deterministic
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

def build_split_lists(images_root: Path, class_to_index: Dict[str,int] | None = None) -> Tuple[SplitLists, Dict[str,int]]:
    """
    IMPORTANT: Mirrors the dynamic trainer's split logic:
      - One rng = random.Random(SEED) created once
      - rng.shuffle(rec_dirs) per species (shared RNG state across species)
    This ensures identical TRAIN/VAL/TEST partitions. (See train_resnet_minority_aug_dynamic.py)
    """
    # If class_to_index provided (from checkpoint), enforce same label ordering.
    if class_to_index is not None:
        species = sorted(class_to_index.keys(), key=lambda s: class_to_index[s])
    else:
        species = list_species(images_root)
        class_to_index = {sp: i for i, sp in enumerate(species)}

    train_items: List[Tuple[Path,int]] = []
    val_items:   List[Tuple[Path,int]] = []
    test_items:  List[Tuple[Path,int]] = []

    rng = random.Random(SEED)  # single RNG, reused across species

    for sp in species:
        sp_dir = images_root / sp
        if not sp_dir.exists():
            print(f"[warn] species folder missing: {sp_dir}")
            continue
        rec_dirs = list_records(sp_dir)
        rng.shuffle(rec_dirs)  # shared RNG state -> matches training script behavior

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

        print(f"[{sp}] recs total={n} -> train={n_tr}, val={n_v}, test={n_te}")

    return SplitLists(train_items, val_items, test_items), class_to_index

# --------- dataset ---------
class ListImageDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int]]):
        self.items = items
        self.to_tensor = transforms.Compose([
            transforms.PILToTensor(),                 # uint8 -> (C,H,W)
            transforms.ConvertImageDtype(torch.float32),  # -> float [0,1]
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with open(path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            img = img.convert("RGB")  # be safe about single-channel
        x = self.to_tensor(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# --------- model ---------
def build_model(num_classes: int) -> nn.Module:
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# --------- metrics ---------
def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_metrics(cm: np.ndarray):
    # cm[r, c] = count of true class r predicted as c
    tp = np.diag(cm).astype(np.float64)
    fp = (cm.sum(axis=0) - tp)
    fn = (cm.sum(axis=1) - tp)
    # precision, recall, f1 (safe divide)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp) > 0)
    rec  = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn) > 0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec) > 0)
    support = cm.sum(axis=1)
    return prec, rec, f1, support

def overall_accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(1, cm.sum()))

def save_cm_csv(cm: np.ndarray, class_names: List[str], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + list(map(int, row)))

def save_cm_png(cm: np.ndarray, class_names: List[str], path: Path, title: str):
    fig = plt.figure(figsize=(max(6, len(class_names)*0.5), max(5, len(class_names)*0.5)))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# --------- evaluation ---------
@torch.no_grad()
def run_split(name: str, model: nn.Module, items: List[Tuple[Path,int]], class_names: List[str]):
    print(f"\n=== Evaluating {name} ({len(items)} images) ===")
    ds = ListImageDataset(items)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0
    )

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    model.eval()
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred_all.extend(preds.tolist())
        y_true_all.extend(yb.numpy().tolist())

    y_true = np.array(y_true_all, dtype=np.int64)
    y_pred = np.array(y_pred_all, dtype=np.int64)
    num_classes = len(class_names)

    cm = confusion_matrix(num_classes, y_true, y_pred)
    acc = overall_accuracy(cm)

    prec, rec, f1, sup = per_class_metrics(cm)

    # Print summary
    print(f"[{name}]  accuracy: {acc*100:.2f}%  (correct {np.trace(cm)}/{cm.sum()})")
    print(f"[{name}]  classes: {num_classes}")
    print(f"[{name}]  per-class metrics:")
    colw = max(8, max(len(c) for c in class_names))
    header = f"{'class':{colw}}  {'support':>8}  {'precision':>9}  {'recall':>7}  {'f1':>6}"
    print(header)
    for i, cname in enumerate(class_names):
        print(f"{cname:{colw}}  {int(sup[i]):8d}  {prec[i]:9.3f}  {rec[i]:7.3f}  {f1[i]:6.3f}")

    # Save artifacts
    safe_name = name.lower().replace("/", "_").replace(" ", "_")
    cm_csv = OUT_DIR / f"confusion_{safe_name}.csv"
    cm_png = OUT_DIR / f"confusion_{safe_name}.png"
    save_cm_csv(cm, class_names, cm_csv)
    save_cm_png(cm, class_names, cm_png, title=f"Confusion Matrix — {name}")

    print(f"[{name}]  saved confusion CSV -> {cm_csv}")
    print(f"[{name}]  saved confusion PNG -> {cm_png}")

    return {
        "name": name,
        "accuracy": acc,
        "cm": cm,
        "per_class": {
            "precision": prec, "recall": rec, "f1": f1, "support": sup
        }
    }

def main():
    # --------- load checkpoint (best preferred, fallback to last) ---------
    best_path = OUT_DIR / "best.pt"
    last_path = OUT_DIR / "last.pt"
    ckpt_path = best_path if best_path.exists() else last_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {best_path} or {last_path}")

    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # class mapping
    if "class_to_index" not in ckpt:
        raise KeyError("Checkpoint missing 'class_to_index'. Re-save best.pt with this field.")
    class_to_index: Dict[str,int] = ckpt["class_to_index"]
    # index -> class for ordered names
    index_to_class = {i: c for c, i in class_to_index.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    # model
    num_classes = len(class_names)
    model = build_model(num_classes)
    state_key = "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key], strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"[model] resnet18 with {num_classes} classes on {DEVICE}")

    # --------- rebuild split (identical to training) ---------
    splits, _ = build_split_lists(IMAGES_ROOT, class_to_index=class_to_index)

    # --------- evaluate per split ---------
    results = []
    results.append(run_split("TRAIN", model, splits.train, class_names))
    results.append(run_split("VAL",   model, splits.val,   class_names))
    results.append(run_split("TEST",  model, splits.test,  class_names))

    # --------- evaluate ALL images combined ---------
    all_items = splits.train + splits.val + splits.test
    results.append(run_split("ALL", model, all_items, class_names))

    # --------- write a compact summary file ---------
    summary_path = OUT_DIR / "eval_summary.txt"
    with open(summary_path, "w") as f:
        for r in results:
            f.write(f"{r['name']}: accuracy={r['accuracy']*100:.2f}%\n")
    print(f"\n[summary] -> {summary_path}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
