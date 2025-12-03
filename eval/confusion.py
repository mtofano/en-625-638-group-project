#!/usr/bin/env python3
# Render confusion matrices from CSV as clear grid tables (no heatmap),
# with large cells and overlaid counts / row percentages.

from __future__ import annotations
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Constants (edit here)
# =========================
OUT_DIR  = Path("foundations_neural_networks/PetFace/runs/cnn_folder_224_raw_aug_norm")
CM_FILES = [
    OUT_DIR / "confusion_train.csv",
    OUT_DIR / "confusion_val.csv",
    OUT_DIR / "confusion_test.csv",
    OUT_DIR / "confusion_all.csv",
]

FIG_DPI        = 200
BASE_FIGSIZE   = 7.0   # slightly bigger base
ROTATE_X_LABELS = 90
GRID_LINEWIDTH = 1.2   # thicker grid lines
ANN_FS_BASE    = 12    # annotation font base
# =========================


def load_cm_csv(path: Path):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows or len(rows) < 2:
        raise ValueError(f"CSV {path} invalid")

    header = rows[0]
    pred_classes = header[1:]
    true_classes = []
    data = []
    for r in rows[1:]:
        true_classes.append(r[0])
        data.append([int(x) for x in r[1:]])

    cm = np.array(data, dtype=np.int64)
    if len(true_classes) != len(pred_classes) or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"CSV {path} not square")
    return pred_classes, cm


def _auto_sizes(n_classes: int):
    """Figure and font sizes for given number of classes."""
    scale = max(1.0, n_classes / 10.0)
    fig_w = BASE_FIGSIZE * scale
    fig_h = BASE_FIGSIZE * scale
    ann_fs = max(6, int(ANN_FS_BASE / math.sqrt(scale)))
    tick_fs = max(6, int(12 / math.sqrt(scale)))
    return (fig_w, fig_h, ann_fs, tick_fs)


def render_table(cm: np.ndarray,
                 class_names: list[str],
                 title: str,
                 save_path: Path,
                 mode: str = "counts"):
    """
    mode in {"counts", "counts_rowpct"}
    """
    n = len(class_names)
    fig_w, fig_h, ann_fs, tick_fs = _auto_sizes(n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI)

    # draw empty grid (no heatmap)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    # set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=ROTATE_X_LABELS, fontsize=tick_fs)
    ax.set_yticklabels(class_names, fontsize=tick_fs)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=tick_fs + 2, pad=12)

    # draw grid
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="black", linewidth=GRID_LINEWIDTH)
        ax.axvline(i - 0.5, color="black", linewidth=GRID_LINEWIDTH)

    # annotate
    if mode == "counts":
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v != 0:
                    ax.text(j, i, f"{v}", ha="center", va="center", fontsize=ann_fs)
    elif mode == "counts_rowpct":
        rowsum = cm.sum(axis=1, keepdims=True).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.divide(cm, rowsum, out=np.zeros_like(cm, dtype=np.float64), where=rowsum > 0)
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v == 0 and rowsum[i, 0] == 0:
                    continue
                ax.text(j, i, f"{v}\n({pct[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=ann_fs)
    else:
        raise ValueError("mode must be 'counts' or 'counts_rowpct'")

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)


def main():
    for csv_path in CM_FILES:
        if not csv_path.exists():
            print(f"[skip] {csv_path}")
            continue
        class_names, cm = load_cm_csv(csv_path)
        base = csv_path.with_suffix("")
        title_base = base.name.replace("_", " ").title()

        # counts only
        render_table(
            cm, class_names,
            title=f"{title_base} — Counts",
            save_path=base.with_name(base.name + "_counts.png"),
            mode="counts"
        )

        # counts + row %
        render_table(
            cm, class_names,
            title=f"{title_base} — Counts & Row %",
            save_path=base.with_name(base.name + "_counts_rowpct.png"),
            mode="counts_rowpct"
        )

        print(f"[ok] {csv_path.name}: saved *_counts.png and *_counts_rowpct.png")


if __name__ == "__main__":
    main()
