#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Stage-2 trainer (one species per run, chosen via SPECIES constant).

- Pretrained ResNet-18 backbone
- 256→224 transforms (+ color-aware jitter)
- Attribute-aware record split (by primary head: breed/color/color2)
- Per-head class weights (gender=inverse, others=aggressive inverse)
- WeightedRandomSampler on most-imbalanced non-gender head
- Losses: gender=weighted CE; others=focal+weighted CE
- Phase 1 (layer4+heads), Phase 2 (unfreeze layer3)
- Saves checkpoints with the SAME structure your inference expects:
    runs/stage2_{species}_noaug/{species}_best.pt and {species}_last.pt
  (best includes {"model": state_dict, "head_vocabs": {...}})
- Writes evaluation artifacts:
    * confusion_test_<head>.csv
    * confusion_all_<head>.csv
    * eval_summary_stage2.txt  (per-head accuracy / macro-precision/recall/F1)

Species supported (13): cat, chimp, chinchilla, degus, dog, ferret, guineapig,
hamster, hedgehog, javasparrow, parakeet, pig, rabbit
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import io, random, math, os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.models import resnet18

# -----------------------------
# Choose species here (constant)
# -----------------------------
SPECIES = "cat"  # set to one of: cat, chimp, chinchilla, degus, dog, ferret, guineapig, 
#hamster, hedgehog, javasparrow, parakeet, pig, rabbit


# ---- pretrained weights (new/old torchvision compatibility) ----
try:
    from torchvision.models import ResNet18_Weights
    RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
except Exception:
    RESNET18_WEIGHTS = None

# =========================
# GLOBAL CONSTANTS
# =========================
IMAGES_ROOT = Path("foundations_neural_networks/PetFace/images")
ANN_ROOT    = Path("foundations_neural_networks/PetFace/annotations")
RUNS_ROOT   = Path("foundations_neural_networks/PetFace/runs")

SEED          = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

BATCH_SIZE    = 128
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 20
LR_HEADS      = 1e-3
LR_UNFREEZE   = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP       = True
PRINT_EVERY   = 50
CHECKPOINT_EVERY = 1
RESUME        = True
FOCAL_GAMMA_COLOR = 2.0   # if species has color/color1/color2 head(s)
FOCAL_GAMMA_IMBAL = 2.5   # if only breed (very imbalanced)

GENDER_MAP = {"male": 1, "female": 0}
UNKNOWN_TOKENS = {"", "unknown", "unk", "na", "n/a", "nan", None}

rng = random.Random(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# SPECIES CONFIG (heads/CSV)
# Head names must match inference features (e.g., color1/color2 vs color)
# =========================
# Fields:
#   model_class: only used as a name (kept for compatibility)
#   csv: annotation filename
#   heads: order we train in (internal only)
#   attr_specs: head -> {"csv_col": <CSV column>, "type": "gender"|"categorical"}
#   save_vocab: which heads to save in checkpoint "head_vocabs"
#   primary_attr_priority: priority for stratified split / sampler
SPECIES_CONFIG: Dict[str, Dict] = {
    "cat": {
        "model_class": "CatModel",
        "csv": "cat.csv",
        "heads": ["gender", "breed", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"breed": True, "color1": True, "color2": True},
        "primary_attr_priority": ["breed", "color1", "color2"],
    },
    "chimp": {
        "model_class": "ChimpModel",
        "csv": "chimp.csv",
        "heads": ["gender", "breed"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
        },
        "save_vocab": {"breed": True},
        "primary_attr_priority": ["breed"],
    },
    "chinchilla": {
        "model_class": "ChinchillaModel",
        "csv": "chinchilla.csv",
        "heads": ["gender", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"color1": True, "color2": True},
        "primary_attr_priority": ["color1", "color2"],
    },
    "degus": {
        "model_class": "DegusModel",
        "csv": "degus.csv",
        "heads": ["gender", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"color1": True, "color2": True},
        "primary_attr_priority": ["color1", "color2"],
    },
    "dog": {
        "model_class": "DogModel",
        "csv": "dog.csv",
        "heads": ["gender", "breed"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
        },
        "save_vocab": {"breed": True},
        "primary_attr_priority": ["breed"],
    },
    "ferret": {
        "model_class": "FerretModel",
        "csv": "ferret.csv",
        "heads": ["gender", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"color1": True, "color2": True},
        "primary_attr_priority": ["color1", "color2"],
    },
    "guineapig": {
        "model_class": "GuineaPigModel",
        "csv": "guineapig.csv",
        "heads": ["gender", "breed", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"breed": True, "color1": True, "color2": True},
        "primary_attr_priority": ["breed", "color1", "color2"],
    },
    "hamster": {
        "model_class": "HamsterModel",
        "csv": "hamster.csv",
        "heads": ["gender", "breed", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"breed": True, "color1": True, "color2": True},
        "primary_attr_priority": ["breed", "color1", "color2"],
    },
    "hedgehog": {
        "model_class": "HedgehogModel",
        "csv": "hedgehog.csv",
        "heads": ["gender", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"color1": True, "color2": True},
        "primary_attr_priority": ["color1", "color2"],
    },
    "javasparrow": {
        "model_class": "JavaSparrowModel",
        "csv": "javasparrow.csv",
        "heads": ["gender", "color", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "color":  {"csv_col": "Color",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"color": True, "color1": True, "color2": True},
        "primary_attr_priority": ["color", "color1", "color2"],
    },
    "parakeet": {
        "model_class": "ParakeetModel",
        "csv": "parakeet.csv",
        "heads": ["gender", "breed", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"breed": True, "color1": True, "color2": True},
        "primary_attr_priority": ["breed", "color1", "color2"],
    },
    "pig": {
        "model_class": "PigModel",
        "csv": "pig.csv",
        "heads": ["gender", "breed", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"breed": True, "color1": True, "color2": True},
        "primary_attr_priority": ["breed", "color1", "color2"],
    },
    "rabbit": {
        "model_class": "RabbitModel",
        "csv": "rabbit.csv",
        "heads": ["gender", "breed", "color1", "color2"],
        "attr_specs": {
            "gender": {"csv_col": "Gender", "type": "gender"},
            "breed":  {"csv_col": "Breed",  "type": "categorical"},
            "color1": {"csv_col": "Color1", "type": "categorical"},
            "color2": {"csv_col": "Color2", "type": "categorical"},
        },
        "save_vocab": {"breed": True, "color1": True, "color2": True},
        "primary_attr_priority": ["breed", "color1", "color2"],
    },
}

# =========================
# Utilities
# =========================
def list_record_dirs(species_root: Path) -> List[Path]:
    return sorted([p for p in species_root.iterdir() if p.is_dir()])

def list_images(rec_dir: Path) -> List[Path]:
    allow = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"}
    return [p for p in rec_dir.rglob("*") if p.is_file() and p.suffix.lower() in allow]

def compute_sizes(n: int, fracs: Tuple[float,float,float]) -> Tuple[int,int,int]:
    t, v, r = fracs
    s = t+v+r
    t, v, r = t/s, v/s, r/s
    ti, vi, ri = int(n*t), int(n*v), int(n*r)
    rem = n - (ti+vi+ri)
    ti += rem
    return ti, vi, ri

def load_annotations(csv_path: Path, species_cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Name" not in df.columns:
        raise RuntimeError(f"{csv_path.name} missing required column: Name")
    df["Name_str"] = df["Name"].astype(str).str.strip()

    for head, spec in species_cfg["attr_specs"].items():
        col = spec["csv_col"]
        if col not in df.columns:
            raise RuntimeError(f"{csv_path.name} missing required column: {col} for head '{head}'")
        if spec["type"] == "gender":
            df[f"{col}_norm"] = df[col].astype(str).str.strip().str.lower()
        else:
            df[f"{col}_norm"] = df[col].astype(str).str.strip()
    return df

def build_vocab(df: pd.DataFrame, head: str, species_cfg: Dict) -> List[str]:
    col = species_cfg["attr_specs"][head]["csv_col"]
    vals = []
    for v in df[f"{col}_norm"].unique().tolist():
        vs = "" if v is None else str(v).strip()
        if vs.lower() in UNKNOWN_TOKENS or vs == "":
            continue
        vals.append(vs)
    return sorted(set(vals))

def to_id(head: str, value: str, vocab: Optional[List[str]], species_cfg: Dict) -> Optional[int]:
    spec = species_cfg["attr_specs"][head]
    if spec["type"] == "gender":
        gl = (value or "").strip().lower()
        if gl in UNKNOWN_TOKENS: return None
        return GENDER_MAP.get(gl, None)
    else:
        vs = (value or "").strip()
        if vs.lower() in UNKNOWN_TOKENS or vs == "": return None
        try:
            return vocab.index(vs) if vocab is not None else None
        except ValueError:
            return None

@dataclass
class Sample:
    path: Path
    record_id: str
    targets: Dict[str, Optional[int]]

@dataclass
class Split:
    train: List[Sample]
    val:   List[Sample]
    test:  List[Sample]

def build_samples(species: str, df: pd.DataFrame, vocabs: Dict[str, Optional[List[str]]], species_cfg: Dict) -> List[Sample]:
    species_root = IMAGES_ROOT / species
    rec_dirs = list_record_dirs(species_root)
    name_to_row = {row["Name_str"]: row for _, row in df.iterrows()}
    samples: List[Sample] = []
    missing = 0

    for rec in rec_dirs:
        rec_id = rec.name
        row = name_to_row.get(rec_id)
        if row is None:
            try: row = name_to_row.get(str(int(rec_id)))
            except Exception: pass
        if row is None: row = name_to_row.get(rec_id.lstrip("0"))
        if row is None: row = name_to_row.get(rec_id.lower())

        t: Dict[str, Optional[int]] = {}
        if row is None:
            missing += 1
            for head in species_cfg["heads"]: t[head] = None
        else:
            for head in species_cfg["heads"]:
                col = species_cfg["attr_specs"][head]["csv_col"]
                norm = row[f"{col}_norm"]
                t[head] = to_id(head, norm, vocabs.get(head), species_cfg)

        for ip in list_images(rec):
            samples.append(Sample(path=ip, record_id=rec_id, targets=t.copy()))

    if missing > 0:
        print(f"[warn] {missing} {species} record folders had no row in {species_cfg['csv']}")

    print("[labels]", end=" ")
    for head in species_cfg["heads"]:
        known = sum(1 for s in samples if s.targets[head] is not None)
        print(f"{head} {known}/{len(samples)}", end="  ")
    print()
    return samples

def split_by_record(samples: List[Sample], species_cfg: Dict) -> Split:
    rec_to_samples: Dict[str, List[Sample]] = {}
    for s in samples:
        rec_to_samples.setdefault(s.record_id, []).append(s)

    primary = None
    for h in species_cfg["primary_attr_priority"]:
        if h in species_cfg["heads"] and h != "gender":
            primary = h; break

    rec_primary: Dict[str, Optional[int]] = {}
    for rec_id, rec_samps in rec_to_samples.items():
        if primary is None:
            rec_primary[rec_id] = None
            continue
        counts: Dict[int,int] = {}
        for s in rec_samps:
            v = s.targets[primary]
            if v is not None: counts[v] = counts.get(v, 0) + 1
        rec_primary[rec_id] = max(counts, key=counts.get) if counts else None

    class_to_recs: Dict[int, List[str]] = {}
    unknown: List[str] = []
    for rec_id, c in rec_primary.items():
        if c is None: unknown.append(rec_id)
        else: class_to_recs.setdefault(c, []).append(rec_id)

    rec_tr, rec_v, rec_te = set(), set(), set()
    for c, rec_ids in class_to_recs.items():
        r = list(rec_ids); rng.shuffle(r)
        n_tr, n_v, n_te = compute_sizes(len(r), (TRAIN_FRAC, VAL_FRAC, TEST_FRAC))
        rec_tr.update(r[:n_tr]); rec_v.update(r[n_tr:n_tr+n_v]); rec_te.update(r[n_tr+n_v:])

    if unknown:
        r = list(unknown); rng.shuffle(r)
        n_tr, n_v, n_te = compute_sizes(len(r), (TRAIN_FRAC, VAL_FRAC, TEST_FRAC))
        rec_tr.update(r[:n_tr]); rec_v.update(r[n_tr:n_tr+n_v]); rec_te.update(r[n_tr+n_v:])

    tr, va, te = [], [], []
    for s in samples:
        if s.record_id in rec_tr: tr.append(s)
        elif s.record_id in rec_v: va.append(s)
        else: te.append(s)

    print(f"[split] records total={len(rec_to_samples)} -> train={len(rec_tr)}, val={len(rec_v)}, test={len(rec_te)}")
    print(f"[split] images  train={len(tr)}, val={len(va)}, test={len(te)}")
    return Split(tr, va, te)

def make_transforms(species_cfg: Dict):
    hue_j = 0.01 if any(h in species_cfg["heads"] for h in ("color","color1","color2")) else 0.02
    EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    TRAIN_TF = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=hue_j),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
    ])
    return TRAIN_TF, EVAL_TF

class SpeciesDataset(Dataset):
    def __init__(self, items: List[Sample], heads: List[str], tf: T.Compose):
        self.items = items
        self.heads = heads
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        s = self.items[idx]
        with open(s.path, "rb") as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        x = self.tf(img)
        ys = [(-1 if s.targets[h] is None else int(s.targets[h])) for h in self.heads]
        return x, torch.tensor(ys, dtype=torch.long)

class _MultiHeadModel(nn.Module):
    def __init__(self, heads: List[str], num_classes: Dict[str,int]):
        super().__init__()
        m = resnet18(weights=RESNET18_WEIGHTS) if RESNET18_WEIGHTS is not None else resnet18(pretrained=True)
        self.backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        feat_dim = m.fc.in_features
        self.heads = nn.ModuleDict({h: nn.Linear(feat_dim, num_classes[h]) for h in heads})
        self._head_order = heads
    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return tuple(self.heads[h](f) for h in self._head_order)

def masked_ce(logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor]=None):
    mask = targets >= 0
    if mask.any():
        return nn.functional.cross_entropy(logits[mask], targets[mask], weight=weight, reduction="mean")
    return logits.sum() * 0.0

def masked_focal_ce(logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor]=None, gamma: float=2.0):
    mask = targets >= 0
    if not mask.any():
        return logits.sum() * 0.0
    logits_m = logits[mask]
    targets_m = targets[mask]
    log_probs = nn.functional.log_softmax(logits_m, dim=1)
    probs = log_probs.exp()
    idx = torch.arange(targets_m.shape[0], device=logits_m.device)
    true_log_probs = log_probs[idx, targets_m]
    true_probs = probs[idx, targets_m]
    focal = (1.0 - true_probs) ** gamma
    loss = -focal * true_log_probs
    if weight is not None:
        cls_w = weight[targets_m]
        loss = loss * cls_w
    return loss.mean()

@torch.no_grad()
def head_acc(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float,int]:
    mask = targets >= 0
    n = int(mask.sum().item())
    if n == 0: return float("nan"), 0
    preds = logits.argmax(1)
    return (preds[mask] == targets[mask]).float().mean().item(), n

def inverse_freq_weights(counts: List[int], aggressive: bool, alpha: float=0.7) -> torch.Tensor:
    pos = [c for c in counts if c>0]
    if not pos:
        return torch.tensor([1.0]*len(counts), dtype=torch.float32, device=DEVICE)
    max_c = max(pos)
    ws = []
    for c in counts:
        if c <= 0: w = 1.0
        else:
            w = (max_c/float(c)) ** alpha if aggressive else (max_c/float(c))
        ws.append(max(0.2, min(8.0, w)))
    return torch.tensor(ws, dtype=torch.float32, device=DEVICE)

def counts_by_class(samples: List[Sample], head: str, K: int) -> List[int]:
    cnt = [0]*K
    for s in samples:
        v = s.targets[head]
        if v is not None: cnt[v] += 1
    return cnt

def build_head_weights(train_samples: List[Sample], heads: List[str], num_classes: Dict[str,int]) -> Dict[str, torch.Tensor]:
    w: Dict[str, torch.Tensor] = {}
    for h in heads:
        K = num_classes[h]
        cnt = counts_by_class(train_samples, h, K)
        w[h] = inverse_freq_weights(cnt, aggressive=(h != "gender"), alpha=0.7)
    return w

def sample_weights_for_sampler(train_samples: List[Sample], head: str, class_w: torch.Tensor) -> torch.Tensor:
    weights = []
    for s in train_samples:
        v = s.targets[head]
        w = float(class_w[int(v)].item()) if v is not None else 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.double)

def run_epoch(model, loader, opt, scaler, phase: str, heads: List[str], head_w: Dict[str, torch.Tensor], focal_gamma: float):
    train = (phase == "train")
    model.train(train)
    loss_sum, n_batches = 0.0, 0
    acc_sum = {h:0.0 for h in heads}
    cov_sum = {h:0   for h in heads}

    for step, (xb, y_all) in enumerate(loader, 1):
        xb = xb.to(DEVICE, non_blocking=True)
        y_all = y_all.to(DEVICE, non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP and DEVICE.startswith("cuda")):
            logits = model(xb)  # tuple per head in order
            losses = []
            for i, h in enumerate(heads):
                w = head_w.get(h)
                if h == "gender":
                    losses.append(masked_ce(logits[i], y_all[:, i], weight=w))
                else:
                    losses.append(masked_focal_ce(logits[i], y_all[:, i], weight=w, gamma=focal_gamma))
            loss = torch.stack(losses).sum()

        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for i, h in enumerate(heads):
            acc, cov = head_acc(logits[i], y_all[:, i])
            if not math.isnan(acc): acc_sum[h] += acc
            cov_sum[h] += cov

        loss_sum += loss.item(); n_batches += 1
        if train and step % PRINT_EVERY == 0:
            ms = " ".join([f"{h}_acc={(acc_sum[h]/max(1,n_batches)):.3f}(n~{cov_sum[h]})" for h in heads])
            print(f"[{phase}] step {step} loss={loss.item():.4f} | {ms}")

    out = {"loss": loss_sum/max(1,n_batches)}
    for h in heads:
        out[f"acc_{h}"] = (acc_sum[h]/max(1,n_batches)) if cov_sum[h]>0 else float("nan")
        out[f"cov_{h}"] = cov_sum[h]
    return out

@torch.no_grad()
def build_confusions(model, loader, heads: List[str], num_classes: Dict[str,int]) -> Dict[str, torch.Tensor]:
    model.eval()
    mats = {h: torch.zeros((num_classes[h], num_classes[h]), dtype=torch.int64) for h in heads if num_classes[h] > 0}
    for xb, y_all in loader:
        xb = xb.to(DEVICE)
        y_all = y_all.to(DEVICE)
        logits = model(xb)
        for i, h in enumerate(heads):
            K = num_classes[h]
            if K == 0: continue
            y = y_all[:, i]
            mask = y >= 0
            if not mask.any(): continue
            pred = logits[i].argmax(1)
            for t, p in zip(y[mask].tolist(), pred[mask].tolist()):
                mats[h][t, p] += 1
    return mats

def save_confusions_csv(mats: Dict[str, torch.Tensor], out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, m in mats.items():
        df = pd.DataFrame(m.cpu().numpy())
        df.to_csv(out_dir / f"confusion_{prefix}_{h}.csv", index=False)

def metrics_from_confusion(m: torch.Tensor) -> Dict[str, float]:
    # m: [K,K] rows=true, cols=pred
    M = m.cpu().numpy()
    K = M.shape[0]
    total = M.sum()
    correct = M.trace()
    acc = float(correct) / float(total) if total > 0 else float("nan")
    # per-class
    recalls, precisions, f1s, supports = [], [], [], []
    for c in range(K):
        tp = M[c, c]
        fn = M[c, :].sum() - tp
        fp = M[:, c].sum() - tp
        support = M[c, :].sum()
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1); supports.append(support)
    # macro (ignore classes with zero support)
    valid = [i for i,s in enumerate(supports) if s > 0]
    if len(valid) == 0:
        return {"acc": acc, "macro_prec": float("nan"), "macro_rec": float("nan"), "macro_f1": float("nan")}
    macro_prec = sum(precisions[i] for i in valid) / len(valid)
    macro_rec  = sum(recalls[i]    for i in valid) / len(valid)
    macro_f1   = sum(f1s[i]        for i in valid) / len(valid)
    return {"acc": acc, "macro_prec": macro_prec, "macro_rec": macro_rec, "macro_f1": macro_f1}

def write_eval_summary(out_dir: Path, mats_test: Dict[str, torch.Tensor], mats_all: Dict[str, torch.Tensor], heads: List[str], vocabs: Dict[str, Optional[List[str]]]):
    lines = []
    lines.append("=== Stage-2 Evaluation Summary ===")
    for split_name, mats in [("TEST", mats_test), ("ALL", mats_all)]:
        lines.append(f"\n[{split_name}]")
        for h in heads:
            if h not in mats: continue
            m = mats[h]
            met = metrics_from_confusion(m)
            K = m.shape[0]
            lines.append(f"- {h}: acc={met['acc']:.4f}  macro_prec={met['macro_prec']:.4f}  macro_rec={met['macro_rec']:.4f}  macro_f1={met['macro_f1']:.4f}  (K={K})")
            # Optional: per-class short line (support & recall)
            vocab = vocabs.get(h) or (["female","male"] if h=="gender" else None)
            if vocab and len(vocab) == K:
                row_sums = m.sum(dim=1).tolist()
                diag = torch.diagonal(m).tolist()
                # show top 5 classes by support
                order = sorted(range(K), key=lambda i: row_sums[i], reverse=True)[:min(5, K)]
                for i in order:
                    rec_i = (diag[i] / row_sums[i]) if row_sums[i] > 0 else 0.0
                    lines.append(f"    · {vocab[i]}: support={row_sums[i]}  recall={rec_i:.3f}")
    out_path = out_dir / "eval_summary_stage2.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[eval] wrote {out_path}")

def train_species(species: str):
    assert species in SPECIES_CONFIG, f"Unknown species '{species}'"
    cfg = SPECIES_CONFIG[species]
    heads = cfg["heads"]
    print(f"[info] species={species} heads={heads}")

    out_dir = RUNS_ROOT / f"stage2_{species}_noaug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSV & vocabs
    df = load_annotations(ANN_ROOT / cfg["csv"], cfg)
    vocabs: Dict[str, Optional[List[str]]] = {}
    num_classes: Dict[str, int] = {}
    for h in heads:
        if cfg["attr_specs"][h]["type"] == "gender":
            vocabs[h] = ["female", "male"]  # explicit for checkpoint clarity
            num_classes[h] = 2
        else:
            v = build_vocab(df, h, cfg)
            vocabs[h] = v
            num_classes[h] = len(v)
    print("[vocab sizes]", {h:(len(v) if v is not None else 2) for h,v in vocabs.items()})

    # 2) Samples & split
    samples = build_samples(species, df, vocabs, cfg)
    splits = split_by_record(samples, cfg)

    # 3) Data & transforms
    TRAIN_TF, EVAL_TF = make_transforms(cfg)
    train_ds = SpeciesDataset(splits.train, heads, TRAIN_TF)
    val_ds   = SpeciesDataset(splits.val,   heads, EVAL_TF)
    test_ds  = SpeciesDataset(splits.test,  heads, EVAL_TF)

    # Weights per head
    head_w = build_head_weights(splits.train, heads, num_classes)

    # Sampler on first available non-gender head from priority, with K>0
    primary_sampler_head = None
    for h in cfg["primary_attr_priority"]:
        if h in heads and h != "gender" and num_classes.get(h, 0) > 0:
            primary_sampler_head = h
            break

    if primary_sampler_head is not None:
        sample_w = sample_weights_for_sampler(splits.train, primary_sampler_head, head_w[primary_sampler_head])
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)

    val_loader  = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)

    # 4) Model
    model = _MultiHeadModel(heads, num_classes).to(DEVICE)
    scaler = torch.amp.GradScaler(device="cuda", enabled=USE_AMP and DEVICE.startswith("cuda"))

    # Focal gamma choice
    has_colorish = any(h in ("color","color1","color2") for h in heads)
    focal_gamma = FOCAL_GAMMA_COLOR if has_colorish else FOCAL_GAMMA_IMBAL

    # 5) Phase 1: freeze all but layer4 + heads
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.backbone[-1].parameters(): p.requires_grad = True  # layer4
    for h in heads:
        for p in model.heads[h].parameters(): p.requires_grad = True

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEADS, weight_decay=WEIGHT_DECAY)
    early = EarlyStop(patience=7, delta=1e-4, mode="min")
    best_val = float("inf")
    start_epoch = 0

    ckpt_path = out_dir / f"{species}_last.pt"
    if RESUME and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val = ckpt.get("best_val", best_val)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[resume] loaded epoch {start_epoch-1}, best_val={best_val:.4f}")

    print("[phase 1] heads + layer4")
    for epoch in range(start_epoch, EPOCHS_PHASE1):
        tr = run_epoch(model, train_loader, opt, scaler, "train", heads, head_w, focal_gamma)
        va = run_epoch(model, val_loader,   opt, scaler, "val",   heads, head_w, focal_gamma)

        metrics = " ".join([f"va_acc_{h}={va[f'acc_{h}']:.3f}(n={va[f'cov_{h}']})" for h in heads])
        print(f"[epoch {epoch}] tr_loss={tr['loss']:.4f} va_loss={va['loss']:.4f} | {metrics}")

        if (epoch % CHECKPOINT_EVERY == 0) or (epoch == EPOCHS_PHASE1-1):
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "best_val": best_val}, ckpt_path)
            print(f"[ckpt] saved -> {ckpt_path}")

        if va["loss"] < best_val - 1e-6:
            best_val = va["loss"]
            payload = {"model": model.state_dict(), "head_vocabs": {h: (vocabs[h] or (["female","male"] if h=="gender" else [])) for h in heads}}
            torch.save(payload, out_dir / f"{species}_best.pt")
            print(f"[best] val loss improved to {best_val:.4f}")

        if early.step(va["loss"]):
            print(f"[early stop] stop phase 1 at epoch {epoch}")
            break

    # 6) Phase 2: unfreeze layer3
    print("[phase 2] unfreeze layer3")
    for name, mod in zip(["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4"], model.backbone):
        if name in {"layer3","layer4"}:
            for p in mod.parameters(): p.requires_grad = True

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_UNFREEZE, weight_decay=WEIGHT_DECAY)
    early2 = EarlyStop(patience=5, delta=1e-4, mode="min")

    for epoch in range(EPOCHS_PHASE2):
        tr = run_epoch(model, train_loader, opt, scaler, "train", heads, head_w, focal_gamma)
        va = run_epoch(model, val_loader,   opt, scaler, "val",   heads, head_w, focal_gamma)

        metrics = " ".join([f"va_acc_{h}={va[f'acc_{h}']:.3f}(n={va[f'cov_{h}']})" for h in heads])
        print(f"[ft {epoch}] tr_loss={tr['loss']:.4f} va_loss={va['loss']:.4f} | {metrics}")

        if (epoch % CHECKPOINT_EVERY == 0) or (epoch == EPOCHS_PHASE2-1):
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict()}, ckpt_path)
            print(f"[ckpt] saved -> {ckpt_path}")

        if va["loss"] < best_val - 1e-6:
            best_val = va["loss"]
            payload = {"model": model.state_dict(), "head_vocabs": {h: (vocabs[h] or (["female","male"] if h=="gender" else [])) for h in heads}}
            torch.save(payload, out_dir / f"{species}_best.pt")
            print(f"[best] val loss improved to {best_val:.4f}")

        if early2.step(va["loss"]):
            print(f"[early stop] stop phase 2 at epoch {epoch}")
            break

    # 7) Test + confusions + summary
    print("[test] loading best and evaluating")
    best = torch.load(out_dir / f"{species}_best.pt", map_location="cpu")
    model.load_state_dict(best["model"])

    # Eval loaders
    TRAIN_TF, EVAL_TF = make_transforms(cfg)
    val_loader  = DataLoader(SpeciesDataset(splits.val,   heads, EVAL_TF), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)
    test_loader = DataLoader(SpeciesDataset(splits.test,  heads, EVAL_TF), batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS>0)
    all_loader  = DataLoader(SpeciesDataset(splits.train + splits.val + splits.test, heads, EVAL_TF),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, persistent_workers=NUM_WORKERS>0)

    te = run_epoch(model, test_loader, None, torch.amp.GradScaler(device="cuda", enabled=False),
                   "test", heads, build_head_weights(splits.train, heads, num_classes), focal_gamma)
    ms = " ".join([f"acc_{h}={te[f'acc_{h}']:.3f}(n={te[f'cov_{h}']})" for h in heads])
    print(f"[test] loss={te['loss']:.4f} | {ms}")

    mats_test = build_confusions(model, test_loader, heads, num_classes)
    mats_all  = build_confusions(model, all_loader,  heads, num_classes)

    # CSVs named like the original
    save_confusions_csv(mats_test, out_dir, prefix="test")   # confusion_test_<head>.csv
    save_confusions_csv(mats_all,  out_dir, prefix="all")    # confusion_all_<head>.csv

    # human-readable summary
    write_eval_summary(out_dir, mats_test, mats_all, heads, vocabs)

class EarlyStop:
    def __init__(self, patience=7, delta=1e-4, mode="min"):
        self.patience, self.delta, self.mode = patience, delta, mode
        self.best = float("inf") if mode=="min" else -float("inf")
        self.bad = 0
    def step(self, val):
        improved = (val < self.best - self.delta) if self.mode=="min" else (val > self.best + self.delta)
        if improved: self.best = val; self.bad = 0; return False
        self.bad += 1; return self.bad > self.patience

if __name__ == "__main__":
    train_species(SPECIES)
