#!/usr/bin/env python3
"""
Two-stage inference engine for PetFace (fixed layout + pretty print + real names)
---------------------------------------------------------------------------------

- Stage 1: species classifier (ResNet-18) -> predicts species
- Stage 2: species specialist (ResNet-18 + multi-head) -> predicts attributes
  (gender + breed and/or colors) for that species.

Stage-1 checkpoint layout:

  RUNS_ROOT /
    cnn_folder_224_raw_aug /
      STAGE1_CKPT_NAME  (e.g., "best.pt")

Stage-2 checkpoint layout (one specialist per species, species lower-case normalized):

  RUNS_ROOT /
    stage2_<species>_noaug /
      <species>_best.pt

Example:
  runs/cnn_folder_224_raw_aug/best.pt
  runs/stage2_cat_noaug/cat_best.pt
  runs/stage2_dog_noaug/dog_best.pt
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import re
import csv

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms as T
from PIL import Image


# ============================
# CONFIG CONSTANTS (edit here)
# ============================

# Path to an image file (will be resized to 224x224 if needed).
IMAGE_PATH: str | Path = "foundations_neural_networks/PetFace/images/cat/000000/00.png"

# Root folder containing Stage-1 and Stage-2 checkpoints.
RUNS_ROOT: str | Path = "foundations_neural_networks/PetFace/runs"

# Folder containing per-species annotation CSVs: <species>.csv
ANNOTATIONS_DIR: Optional[str | Path] = "foundations_neural_networks/PetFace/annotations"

# Stage-1 checkpoint relative location
STAGE1_SUBDIR: str = "cnn_folder_224_raw_aug"
STAGE1_CKPT_NAME: str = "best.pt"

# Stage-2 checkpoint naming pattern (species will be lower-cased and normalized)
STAGE2_DIR_TEMPLATE: str = "stage2_{species}_noaug"
STAGE2_CKPT_TEMPLATE: str = "{species}_best.pt"

# Top-k species to report from Stage-1.
TOPK_SPECIES: int = 5

# Top-k per head to report from Stage-2.
TOPK_HEAD: int = 5

# Force device (None = auto; or "cpu" / "cuda").
DEVICE: Optional[str] = None

# Optional: write JSON output here (no JSON printed to terminal).
JSON_OUT: Optional[str | Path] = None


# ============================
# Hard-coded features per species (semantic names)
# ============================

SPECIES_FEATURES: Dict[str, List[str]] = {
    "cat":        ["breed", "color1", "color2", "gender"],
    "chimp":      ["gender", "breed"],
    "chinchilla": ["color1", "gender", "color2"],
    "degus":      ["gender", "color1", "color2"],
    "dog":        ["breed", "gender"],
    "ferret":     ["gender", "color1", "color2"],
    "guineapig":  ["gender", "breed", "color1", "color2"],
    "hamster":    ["gender", "breed", "color1", "color2"],
    "hedgehog":   ["gender", "color1", "color2"],
    "javasparrow":["color", "gender", "color1", "color2"],
    "parakeet":   ["gender", "breed", "color1", "color2"],
    "pig":        ["gender", "breed", "color1", "color2"],
    "rabbit":     ["gender", "breed", "color1", "color2"],
}

# For some species, the stage-2 template used a generic "attr2" head.
# This maps that generic head name to the real semantic one.
ATTR2_SEMANTIC_BY_SPECIES: Dict[str, str] = {
    "cat":        "breed",
    "dog":        "breed",
    "parakeet":   "breed",
    "javasparrow":"color",
    # you can extend this dict for any other attr2-based models
}


# ============================
# Hard-coded breed vocabs by species (from annotations)
# ============================

BREED_VOCABS_BY_SPECIES: Dict[str, List[str]] = {
    'cat': [
        'Abyssinian',
        'Abyssinian Mix',
        'American Curl',
        'American Curl Mix',
        'American Shorthair',
        'American Shorthair Mix',
        'Bengal',
        'Bengal Mix',
        'Birman',
        'Bombay',
        'British Longhair',
        'British Shorthair',
        'Burmese',
        'Chartreux',
        'Cornish Rex',
        'Devon Rex',
        'Domestic Long Hair',
        'Domestic Medium Hair',
        'Domestic Short Hair',
        'Domestic Short Hair Mix',
        'Exotic Shorthair',
        'Himalayan',
        'Himalayan Mix',
        'Japanese Bobtail',
        'Japanese Bobtail Mix',
        'Maine Coon',
        'Maine Coon Mix',
        'Manx',
        'Manx Mix',
        'Mixed Breed',
        'Munchkin',
        'Munchkin Mix',
        'Norwegian Forest',
        'Norwegian Forest Mix',
        'Oriental Long Hair',
        'Oriental Long Hair Mix',
        'Oriental Short Hair',
        'Oriental Short Hair Mix',
        'Persian',
        'Persian Mix',
        'Peterbald Mix',
        'Ragamuffin',
        'Ragdoll',
        'Ragdoll Mix',
        'Russian Blue',
        'Russian Blue Mix',
        'Scottish Fold',
        'Siamese',
        'Siamese Mix',
        'Siberian',
        'Siberian Mix',
        'Somali',
        'Tonkinese',
        'Tonkinese Mix',
        'Turkish Angora',
        'Turkish Van',
    ],
    'chimp': [
        'Mixed Breed',
        'P. t. ellioti',
        'P. t. schweinfurthii',
        'P. t. troglodytes',
        'P. t. verus',
    ],
    'dog': [
        'Airedale Terrier',
        'Akita',
        'Akita and German Shepherd mix',
        'Alsatian',
        'American Staffordshire Terrier',
        'American pitbull terrier (APBT)',
        'Australian Cattle Dog',
        'Australian Shepard',
        'Australian Shepard mix',
        'Australian Terrier',
        'Basenji',
        'Basset Hound',
        'Beagle',
        'Bearded Collie',
        'Belgian Malinois',
        'Belgian Shepherd mix',
        'Bishon Frise',
        'Bloodhound',
        'Blue Heeler',
        'Bluetick',
        'Border Collie',
        'Border Collie mix',
        'Border Terrier',
        'Boston Terrier',
        'Boxer',
        'Boxer Mix',
        'Bull Terrier',
        'Bulldog',
        'Bulldog, Olde English Playground Bulldog mix',
        'Bullmastiff',
        'Cane Corso',
        'Cavalier King Charles Spaniel',
        'Chihuahua',
        'Chihuahua Mix',
        'Chinese Crested Hairless',
        'Chow Chow',
        'Cockapoo',
        'Cocker Spaniel',
        'Collie',
        'Coonhound',
        'Corgi',
        'Coton de Tulear',
        'Dachshund',
        'Dalmatian',
        'Doberman',
        'English Coonhound',
        'English Pointer, German Shorthaired Pointer, Whippet mix',
        'English Setter mix',
        'English Springer Spaniel',
        'English Toy Spaniel',
        'Field Spaniel',
        'French Bulldog',
        'German Shepherd',
        'German Shepherd mix',
        'German Shorthaired Pointer',
        'Giant Schnauzer',
        'Goldendoodle',
        'Golden Retriever',
        'Golden Retriever mix',
        'Gordon Setter',
        'Great Dane',
        'Great Pyrenees',
        'Greyhound',
        'Havanese',
        'Hound mixed breed',
        'Icelandic Sheepdog mix',
        'Labrador Retriever',
        'Labrador Retriever mix',
        'Labrador mix dog',
        'Lhasa',
        'Maltese',
        'Mastiff',
        'Mastiff mix',
        'Mixed Breed',
        'Mixed breed, pitbull terrier type',
        'Newfoundland',
        'Norfolk Terrier mix',
        'Norwegian Elkhound',
        'Papillon',
        'Pekinese',
        'Pembroke Welsh Corgi',
        'Pitbull',
        'Pitbull terrier type',
        'Pitbull/X  Mixed Breed',
        'Pointer and cow or mixed breed',
        'Pomchi',
        'Pomeranian',
        'Pomeranian mixed breed',
        'Poodle',
        'Poodle mix',
        'Portuguese Water Dog',
        'Pug',
        'Rat Terrier',
        'Rhodesian Ridgeback',
        'Rottweiler',
        'Saint Bernard',
        'Samoyed',
        'Schnauzer',
        'Schnauzer mix',
        'Scottish Terrier',
        'Setter',
        'Sharpei',
        'Sheltie',
        'Sheltie mix, Rough Collie mix',
        'Sheltie mix, spaniel mix',
        'Shetland Sheepdog',
        'Shiba Inu',
        'Shih Tzu',
        'Shih Tzu mix',
        'Smooth collie',
        'Spanish water dog',
        'Springer Spaniel',
        'Staffordshire Terrier mix',
        'Standard Poodle and Flat-Coated Retriever mix',
        'Suffolk Sheepdog mix',
        'Terrier mix',
        'Toy Poodle, Bichon Frise mix',
        'Toy- or miniature poodle mix',
        'Tree Walker Coonhound',
        'Vizsla',
        'Weimaraner',
        'Welsh Terrier mix',
        'West Highland White Terrier',
        'Wheaten Terrier',
        'Whippet',
        'Yorkshire Terrier',
        'Yorkshire Terrier Mix',
    ],
    'guineapig': [
        'Abbysinian',
        'Abbysinian Mix',
        'Abbysinian Peruvian mix',
        'American Satin',
        'American mix',
        'American short hair',
        'Crested Pig',
        'Mixed Breed',
        'Peruvian',
        'Peruvian Mix',
        'Silkie',
        'Silkie mix',
        'Teddy',
        'Teddy mix',
        'Texel',
        'Unknown Longhair',
    ],
    'hamster': [
        'Chinese Hamster',
        'Dwarf Hamster',
        'Fancy Hamster mix',
        'Russian hamster',
        'Syrian Hamster',
    ],
    'parakeet': [
        'American Parakeet',
        'American Parakeet mix',
        'Bourkes Parakeet',
        'Budgie mix',
        'English Budgie',
        'English Budgie mix',
        'Grass Parakeet',
        'Half sider',
        'Quaker Parrot',
    ],
    'pig': [
        'micro pig',
        'standard size pig',
    ],
    'rabbit': [
        'American Fuzzy Lop',
        'Angora',
        'Dwarf Hotot',
        'Flemish Giant',
        'French Lop',
        'Himalayan',
        'Holland Lop',
        'Lion Lop Ear',
        'Lion Rabbit',
        'Lop Ear Rabbit',
        'Mini Rabbit',
        'Mini Rex',
        'Mixed Breed',
        'Netherland Dwarf',
        'Polish',
        'Rex',
    ],
}

# ============================
# Hard-coded Color1 / Color2 / Color vocabs per species
# ============================

COLOR1_VOCABS_BY_SPECIES: Dict[str, List[str]] = {
    'cat': [
        'Black',
        'Calico',
        'Gray',
        'Pied',
        'Seal',
        'Tabby',
        'Tortoiseshell',
        'White',
    ],
    'chinchilla': [
        'Black',
        'Cinnamon',
        'Gray',
        'Orange',
        'Pied',
        'Violet',
        'White',
    ],
    'degus': [
        'Agouti',
        'Black',
        'Blue',
        'Chocolate',
        'Pied',
        'Sand',
        'Silver',
        'White',
        'Yellow',
    ],
    'ferret': [
        'Albino',
        'Champagne',
        'Cinnamon',
        'Sable',
        'Silver',
    ],
    'guineapig': [
        'Albino',
        'Beige',
        'Black',
        'Brown',
        'Gray',
        'Pied',
        'Tricolor',
        'White',
    ],
    'hamster': [
        'Albino',
        'Black',
        'Cream',
        'Golden',
        'Gray',
        'Pied',
        'Silver',
        'White',
    ],
    'hedgehog': [
        'Albino',
        'Cinnamon',
        'Gray',
    ],
    'javasparrow': [
        'Black',
        'Cinnamon',
        'Pied',
        'Silver',
        'White',
    ],
    'parakeet': [
        'Blue',
        'Gray',
        'Pied',
        'White',
        'Yellow',
    ],
    'pig': [
        'Black',
        'Caramel',
        'Cream',
        'Ginger',
        'Gray',
        'Pied',
        'Spotted',
        'White',
    ],
    'rabbit': [
        'Black',
        'Brown',
        'Chestnut',
        'Gray',
        'Orange',
        'Pied',
        'White',
    ],
}

COLOR2_VOCABS_BY_SPECIES: Dict[str, List[str]] = {
    'cat': [
        'BlackandWhite',
        'Brown',
        'Gray',
        'Orange',
        'Point',
    ],
    'chinchilla': [
        'Black and White',
        'Gray and White',
        'Violet and White',
    ],
    'degus': [
        'Agouti',
        'Black',
        'Blue',
        'Sand',
        'Silver',
        'White',
        'Yellow',
    ],
    'ferret': [
        # no valid Color2 tokens (all unknown/empty) in annotations
    ],
    'guineapig': [
        'Black and White',
        'Brown and Black',
        'Brown and Gray',
        'Brown and White',
        'Gray and Beige',
        'Gray and White',
    ],
    'hamster': [
        'Black and Golden',
        'Black and White',
        'Golden and Gray',
        'Golden and White',
        'Gray and White',
        'Silver and White',
    ],
    'hedgehog': [
        # no valid Color2 tokens in annotations
    ],
    'javasparrow': [
        'Black',
        'Silver',
        'White',
    ],
    'parakeet': [
        'Blue',
        'Gray',
        'Green',
        'Pink',
        'Yellow',
    ],
    'pig': [
        'Black',
        'Caramel',
        'Cream',
        'Gray',
        'White',
    ],
    'rabbit': [
        'BlackandWhite',
        'BrownandWhite',
        'ChestnutandWhite',
        'GrayandWhite',
        'OrangeandBlack',
        'OrangeandWhite',
    ],
}

# Some species also have a base "Color" feature (e.g. javasparrow)
COLOR_VOCABS_BY_SPECIES: Dict[str, List[str]] = {
    'javasparrow': ['Cinnamon', 'Pink', 'Silver', 'White'],
}


# ============================
# Helper functions / transforms
# ============================

UNKNOWN_TOKENS = {"", "unknown", "unk", "na", "n/a", "nan", None}

def build_eval_transform(resize_to: Tuple[int, int] = (224, 224)) -> T.Compose:
    return T.Compose([
        T.Resize(resize_to, antialias=True),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
    ])


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _load_ckpt(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Try to pull a proper state_dict out of a checkpoint that may store:
      - a plain dict of tensors,
      - a 'model' key holding a dict or an nn.Module,
      - a 'state_dict' key holding a dict or an nn.Module,
      - or tensors at the top level.
    """
    # 1) ckpt["model"] can be dict OR nn.Module
    if "model" in ckpt:
        m = ckpt["model"]
        if isinstance(m, dict):
            return m
        if isinstance(m, nn.Module):
            return m.state_dict()

    # 2) ckpt["state_dict"] can be dict OR nn.Module
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        if isinstance(sd, dict):
            return sd
        if isinstance(sd, nn.Module):
            return sd.state_dict()

    # 3) Maybe the checkpoint itself is already a state_dict-like mapping
    keys = [k for k in ckpt.keys() if isinstance(ckpt[k], torch.Tensor)]
    if keys:
        return ckpt

    raise KeyError("No model/state_dict found in checkpoint.")

def build_attr2_vocab_from_csv(
    annotations_dir: Optional[Path],
    species: str,
    alias: str,
) -> Optional[List[str]]:
    """
    Rebuild attr2 vocab from annotations/<species>.csv using same logic as training:

        - col_name = "Breed" if alias == "breed" else "Color" if alias == "color"
        - strip, drop UNKNOWN_TOKENS / empty, sort unique.
    """
    if annotations_dir is None:
        return None
    csv_path = annotations_dir / f"{_norm(species)}.csv"
    if not csv_path.exists():
        return None

    if alias == "breed":
        col_name = "Breed"
    elif alias == "color":
        col_name = "Color"
    else:
        return None

    vals: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if col_name not in (reader.fieldnames or []):
            return None
        for row in reader:
            raw = row.get(col_name, "")
            vs = "" if raw is None else str(raw).strip()
            if _norm(vs) in UNKNOWN_TOKENS or vs == "":
                continue
            vals.append(vs)
    if not vals:
        return None
    # sorted unique, like build_attr2_vocab
    vocab = sorted(set(vals))
    return vocab


# ----------------------------
# Stage-1: species classifier
# ----------------------------

def get_stage1_ckpt_path(runs_root: Path) -> Path:
    path = runs_root / STAGE1_SUBDIR / STAGE1_CKPT_NAME
    if not path.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {path}")
    return path


@dataclass
class Stage1Model:
    model: nn.Module
    class_to_index: Dict[str, int]
    device: str
    tf: T.Compose

    @classmethod
    def load_from_runs(
        cls,
        runs_root: Path,
        device: Optional[str] = None,
        resize_to: Tuple[int, int] = (224, 224),
    ) -> "Stage1Model":
        ckpt_path = get_stage1_ckpt_path(runs_root)
        ckpt = _load_ckpt(ckpt_path)
        if "class_to_index" not in ckpt:
            raise KeyError(f"'class_to_index' not found in Stage-1 checkpoint: {ckpt_path}")
        class_to_index = ckpt["class_to_index"]
        num_classes = len(class_to_index)

        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        sd = _extract_state_dict(ckpt)
        m.load_state_dict(sd, strict=True)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m.to(dev)
        m.eval()
        return cls(
            model=m,
            class_to_index=class_to_index,
            device=dev,
            tf=build_eval_transform(resize_to),
        )

    @torch.no_grad()
    def predict(self, img: Image.Image, topk: int = 5) -> Dict[str, Any]:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()

        idx_to_class = {i: c for c, i in self.class_to_index.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        topk = min(topk, len(class_names))
        pvals, idxs = torch.topk(probs, k=topk)

        result = [
            {"label": class_names[int(i)], "p": float(p)}
            for p, i in zip(pvals.tolist(), idxs.tolist())
        ]
        pred_label = class_names[int(torch.argmax(probs).item())]
        return {"label": pred_label, "probs": result}


# --------------------------------
# Stage-2: specialist (multi-head)
# --------------------------------

def _headname_from_key(k: str) -> Optional[str]:
    m = re.match(r"^head_([a-zA-Z0-9_]+)\.(weight|bias)$", k)
    return m.group(1) if m else None


def _infer_head_defs_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    heads: Dict[str, int] = {}
    for k, v in state.items():
        name = _headname_from_key(k)
        if name and k.endswith(".weight") and v.ndim == 2:
            heads[name] = v.shape[0]
    return heads


class GenericMultiHeadResNet18(nn.Module):
    """
    ResNet-18 backbone + dynamic heads, matching the training layout in the *_resnet.py scripts.
    """
    def __init__(self, heads: Dict[str, int]):
        super().__init__()
        m = resnet18(weights=None)
        self.backbone = nn.Sequential(
            m.conv1,   # 0
            m.bn1,     # 1
            m.relu,    # 2
            m.maxpool, # 3
            m.layer1,  # 4
            m.layer2,  # 5
            m.layer3,  # 6
            m.layer4,  # 7
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = m.fc.in_features
        for hname, k in heads.items():
            setattr(self, f"head_{hname}", nn.Linear(feat_dim, int(k)))

    def forward(self, x) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        out: Dict[str, torch.Tensor] = {}
        for name, mod in self.named_children():
            if name.startswith("head_"):
                out[name[5:]] = mod(f)
        return out

    def load_state_dict_strict(self, state: Dict[str, torch.Tensor]):
        incompat = self.load_state_dict(state, strict=True)
        return list(incompat.missing_keys), list(incompat.unexpected_keys)

class ChinchillaMultiHeadResNet18(nn.Module):
    """
    Match the training structure in chinchilla_resnet.py:

        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.heads = nn.ModuleDict({feat_name: nn.Linear(feat_dim, K_feat)})

    The checkpoint stores weights in keys like:
        "backbone.0.weight", ..., "heads.Gender.weight", "heads.Color1.weight", ...
    """
    def __init__(self, feature_vocabs: Dict[str, List[str]]):
        super().__init__()
        m = resnet18(weights=None)
        self.backbone = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        feat_dim = m.fc.in_features

        heads = {}
        for feat_name, vocab in feature_vocabs.items():
            K = len(vocab)                  # allow K == 0 to match training
            heads[feat_name] = nn.Linear(feat_dim, K)

        self.heads = nn.ModuleDict(heads)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        out: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            out[name] = head(f)
        return out


def get_stage2_ckpt_path(runs_root: Path, species_label: str) -> Optional[Path]:
    sp = _norm(species_label)
    if not sp:
        return None
    subdir = STAGE2_DIR_TEMPLATE.format(species=sp)
    fname = STAGE2_CKPT_TEMPLATE.format(species=sp)
    path = runs_root / subdir / fname
    if not path.exists():
        return None
    return path


def _default_vocab_for_head(head: str) -> Optional[List[str]]:
    if _norm(head) == "gender":
        return ["female", "male"]
    return None


@dataclass
class Stage2Model:
    species: str
    model: nn.Module
    head_vocabs: Dict[str, List[str]]   # head-name (normalized) -> list of labels
    device: str
    tf: T.Compose

    @classmethod
    def load_for_species(
        cls,
        runs_root: Path,
        species: str,
        annotations_dir: Optional[Path] = None,
        device: Optional[str] = None,
        resize_to: Tuple[int, int] = (224, 224),
    ) -> Optional["Stage2Model"]:
        sp_norm = _norm(species)

        ckpt_path = get_stage2_ckpt_path(runs_root, species)
        if ckpt_path is None or not ckpt_path.exists():
            return None

        ckpt = _load_ckpt(ckpt_path)

        # -------------------------------------------------
        # 1) New-style ModuleDict layout:
        #    - "feature_infos"  (chinchilla-style)
        #    - "head_vocabs"    (rabbit/cat/etc. multi-head scripts)
        #    - "heads_meta"     (parakeet multi-head script)
        #    - "vocabs"+"heads" (guineapig/hamster/hedgehog scripts)
        #    All correspond to backbone + self.heads = ModuleDict(...)
        # -------------------------------------------------
        feature_vocabs_raw: Optional[Dict[str, List[str]]] = None

        if "feature_infos" in ckpt and isinstance(ckpt["feature_infos"], dict):
            # e.g. chinchilla_resnet.py
            feature_vocabs_raw = ckpt["feature_infos"]

        elif "head_vocabs" in ckpt and isinstance(ckpt["head_vocabs"], dict):
            # e.g. rabbit_resnet.py, cat_resnet.py, etc.
            feature_vocabs_raw = ckpt["head_vocabs"]

        elif "heads_meta" in ckpt and isinstance(ckpt["heads_meta"], list):
            # e.g. parakeet multi-head script
            feature_vocabs_raw = {}
            for h in ckpt["heads_meta"]:
                name = h.get("name")
                htype = h.get("type")
                if not name:
                    continue
                if htype == "gender":
                    vocab = ["female", "male"]
                else:
                    vocab = list(h.get("vocab", []))
                feature_vocabs_raw[name] = vocab

        elif "vocabs" in ckpt and isinstance(ckpt["vocabs"], dict) \
             and "heads" in ckpt and isinstance(ckpt["heads"], (list, tuple)):
            # e.g. guineapig/hamster/hedgehog multi-head scripts
            feature_vocabs_raw = {}
            vocabs_dict = ckpt["vocabs"]
            for name in ckpt["heads"]:
                if name == "gender":
                    # gender head is always 2 classes, not stored in vocabs
                    vocab = ["female", "male"]
                else:
                    vocab = list(vocabs_dict.get(name, []))
                feature_vocabs_raw[name] = vocab



        if feature_vocabs_raw is not None:
            # Build a ModuleDict-headed ResNet-18 that matches training
            mh_model = ChinchillaMultiHeadResNet18(feature_vocabs_raw)
            state = _extract_state_dict(ckpt)  # contains backbone.* and heads.<feat>.*

            # keys should match exactly; strict=True is fine here
            mh_model.load_state_dict(state, strict=True)

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            mh_model.to(dev)
            mh_model.eval()

            # Normalized head -> vocab
            head_vocabs: Dict[str, List[str]] = {
                _norm(feat_name): list(vocab)
                for feat_name, vocab in feature_vocabs_raw.items()
            }

            return cls(
                species=species,
                model=mh_model,
                head_vocabs=head_vocabs,
                device=dev,
                tf=build_eval_transform(resize_to),
            )

        # -------------------------------------------------
        # 2) Legacy "head_*" layout (dog, chimp, etc.)
        # -------------------------------------------------
        state = _extract_state_dict(ckpt)
        head_defs = _infer_head_defs_from_state_dict(state)
        if not head_defs:
            # no recognizable heads in this checkpoint
            return None

        m = GenericMultiHeadResNet18(head_defs)
        _ = m.load_state_dict(state, strict=True)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m.to(dev)
        m.eval()

        vocabs: Dict[str, List[str]] = {}

        # 2.1) Vocab info saved in checkpoint, e.g. "gender_vocab", "breed_vocab"
        for k, v in ckpt.items():
            if isinstance(k, str) and k.endswith("_vocab"):
                headname = k[:-7]  # strip "_vocab"
                if isinstance(v, (list, tuple)):
                    vocabs[_norm(headname)] = list(v)
                else:
                    try:
                        vocabs[_norm(headname)] = list(v)
                    except Exception:
                        pass

        # 2.2) Species-specific hard-coded vocabs
        sp_norm = _norm(species)

        if sp_norm in BREED_VOCABS_BY_SPECIES and "breed" in head_defs:
            breed_vocab = BREED_VOCABS_BY_SPECIES[sp_norm]
            if len(breed_vocab) == head_defs["breed"]:
                if "breed" not in vocabs or len(vocabs["breed"]) != len(breed_vocab):
                    vocabs["breed"] = list(breed_vocab)

        if sp_norm in COLOR_VOCABS_BY_SPECIES and "color" in head_defs:
            c_vocab = COLOR_VOCABS_BY_SPECIES[sp_norm]
            if len(c_vocab) == head_defs["color"]:
                if "color" not in vocabs or len(vocabs["color"]) != len(c_vocab):
                    vocabs["color"] = list(c_vocab)

        if sp_norm in COLOR1_VOCABS_BY_SPECIES and "color1" in head_defs:
            c1 = COLOR1_VOCABS_BY_SPECIES[sp_norm]
            if len(c1) == head_defs["color1"]:
                if "color1" not in vocabs or len(vocabs["color1"]) != len(c1):
                    vocabs["color1"] = list(c1)

        if sp_norm in COLOR2_VOCABS_BY_SPECIES and "color2" in head_defs:
            c2 = COLOR2_VOCABS_BY_SPECIES[sp_norm]
            if len(c2) == head_defs["color2"]:
                if "color2" not in vocabs or len(vocabs["color2"]) != len(c2):
                    vocabs["color2"] = list(c2)

        # 2.3) attr2 semantic mapping (cat/dog/parakeet/javasparrow)
        alias = ATTR2_SEMANTIC_BY_SPECIES.get(sp_norm)
        if "attr2" in head_defs and alias is not None:
            K = head_defs["attr2"]
            if alias == "breed" and sp_norm in BREED_VOCABS_BY_SPECIES:
                breed_vocab = BREED_VOCABS_BY_SPECIES[sp_norm]
                if len(breed_vocab) == K:
                    vocabs["attr2"] = list(breed_vocab)
            elif alias == "color" and sp_norm in COLOR_VOCABS_BY_SPECIES:
                c_vocab = COLOR_VOCABS_BY_SPECIES[sp_norm]
                if len(c_vocab) == K:
                    vocabs["attr2"] = list(c_vocab)

        # 2.4) Defaults (gender)
        for head, K in head_defs.items():
            hn = _norm(head)
            if hn not in vocabs:
                dv = _default_vocab_for_head(hn)
                if dv is not None and len(dv) == K:
                    vocabs[hn] = dv

        # 2.5) Fallback to plain indices
        for head, K in head_defs.items():
            hn = _norm(head)
            if hn not in vocabs or len(vocabs[hn]) != K:
                vocabs[hn] = [str(i) for i in range(K)]

        return cls(
            species=species,
            model=m,
            head_vocabs=vocabs,
            device=dev,
            tf=build_eval_transform(resize_to),
        )



    @torch.no_grad()
    def predict(self, img: Image.Image, topk_per_head: int = 5) -> Dict[str, Any]:
        """
        Run the Stage-2 model on a single PIL image and return per-head predictions.

        Any head whose logits have zero classes (e.g., K=0 for Color2 in ferret)
        is skipped to avoid argmax/softmax on empty tensors.
        """
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits_by_head = self.model(x)
        out: Dict[str, Any] = {}

        for raw_head, logits in logits_by_head.items():
            # logits must be [1, K]; skip heads with K == 0
            if logits.ndim != 2 or logits.size(1) == 0:
                continue

            probs = torch.softmax(logits, dim=1)[0].cpu()  # shape [K]
            K = probs.numel()
            if K == 0:
                continue  # extra guard, though size(1) check should catch this

            head_norm = _norm(raw_head)
            labels = self.head_vocabs.get(head_norm, [str(i) for i in range(K)])
            if len(labels) == 0:
                continue

            topk = min(topk_per_head, len(labels))
            pvals, idxs = torch.topk(probs, k=topk)

            out[raw_head] = {
                "label": labels[int(torch.argmax(probs).item())],
                "probs": [
                    {"label": labels[int(i)], "p": float(p)}
                    for p, i in zip(pvals.tolist(), idxs.tolist())
                ],
            }

        return out




# ----------------------------
# Two-stage pipeline wrapper
# ----------------------------

class TwoStageInference:
    def __init__(
        self,
        runs_root: str | Path,
        annotations_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        resize_to: Tuple[int, int] = (224, 224),
    ) -> None:
        self.runs_root = Path(runs_root)
        self.annotations_dir = Path(annotations_dir) if annotations_dir is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resize_to = resize_to
        self._stage1: Optional[Stage1Model] = None

    def _get_stage1(self) -> Stage1Model:
        if self._stage1 is None:
            self._stage1 = Stage1Model.load_from_runs(
                self.runs_root,
                device=self.device,
                resize_to=self.resize_to,
            )
        return self._stage1

    def infer(
        self,
        image_path: str | Path,
        topk_species: int = 5,
        topk_per_head: int = 5,
    ) -> Dict[str, Any]:
        img = Image.open(image_path).convert("RGB")

        # Stage 1: species
        st1 = self._get_stage1()
        species_pred = st1.predict(img, topk=topk_species)
        species_label = species_pred["label"]
        species_norm = _norm(species_label)

        # Stage 2: attributes
        st2 = Stage2Model.load_for_species(
            self.runs_root,
            species_label,
            annotations_dir=self.annotations_dir,
            device=self.device,
            resize_to=self.resize_to,
        )
        if st2 is None:
            attributes: Dict[str, Any] = {
                "_warning": (
                    f"No Stage-2 checkpoint found for species '{species_label}'. "
                    f"Expected at: {get_stage2_ckpt_path(self.runs_root, species_label)}"
                )
            }
        else:
            attributes = st2.predict(img, topk_per_head=topk_per_head)

            # --- Rename generic attr2 head to semantic name for some species ---
            alias = ATTR2_SEMANTIC_BY_SPECIES.get(species_norm)
            if alias and "attr2" in attributes:
                new_attrs: Dict[str, Any] = {}
                for h, v in attributes.items():
                    if h == "attr2":
                        new_attrs[alias] = v
                    else:
                        new_attrs[h] = v
                attributes = new_attrs

            # Filter to species feature set (breed/gender/colors/etc.)
            wanted_feats = {_norm(f) for f in SPECIES_FEATURES.get(species_norm, [])}
            if wanted_feats:
                attributes = {
                    h: v for h, v in attributes.items()
                    if _norm(h) in wanted_feats
                }

        return {"species": species_pred, "attributes": attributes}


# ----------------------------
# Pretty printing
# ----------------------------

def pretty_print_result(result: Dict[str, Any]) -> None:
    species = result["species"]
    attrs = result["attributes"]

    top_species = species["probs"]
    best = top_species[0] if top_species else {"label": species["label"], "p": 1.0}
    print("=" * 60)
    print(f"Image inference result")
    print("-" * 60)
    print(f"Predicted species: {best['label']}  (p = {best['p']:.4f})")
    print()
    print("Top species probabilities:")
    for i, entry in enumerate(top_species, start=1):
        print(f"  {i:2d}. {entry['label']:15s}  p = {entry['p']:.4f}")
    print()

    if "_warning" in attrs:
        print("Attributes:")
        print(f"  [warning] {attrs['_warning']}")
        print("=" * 60)
        return

    if not attrs:
        print("Attributes: [none predicted]")
        print("=" * 60)
        return

    print("Attributes:")
    for head in sorted(attrs.keys()):
        info = attrs[head]
        label = info["label"]
        probs = info.get("probs", [])
        print(f"  {head}: {label}")
        if probs:
            for j, pentry in enumerate(probs, start=1):
                print(f"    {j:2d}. {pentry['label']:25s}  p = {pentry['p']:.4f}")
        print()
    print("=" * 60)


# ----------------------------
# Main using constants only
# ----------------------------

def main() -> None:
    runs_root = Path(RUNS_ROOT)
    annotations_dir = Path(ANNOTATIONS_DIR) if ANNOTATIONS_DIR is not None else None

    engine = TwoStageInference(
        runs_root=runs_root,
        annotations_dir=annotations_dir,
        device=DEVICE,
    )

    result = engine.infer(
        IMAGE_PATH,
        topk_species=TOPK_SPECIES,
        topk_per_head=TOPK_HEAD,
    )

    pretty_print_result(result)

    if JSON_OUT is not None:
        out_path = Path(JSON_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[saved JSON] {out_path}")


if __name__ == "__main__":
    main()
