"""
check_and_plot_v5.py
====================
Vẽ biểu đồ phân bố số Track và số Frame theo từng class
cho dataset hiện tại (stgcn_augmented + stgcn_split_all).

Output: 4 ảnh PNG (track/frame × augmented/split_all)

Usage:
    python check_and_plot_v5.py
"""

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"d:\Capstone2026\Action Predict\Labeled_data")

DATASETS = {
    "stgcn_augmented": {
        "path":  BASE_DIR / "stgcn_augmented",
        "label": "Augmented",
    },
    "stgcn_split_all": {
        "path":  BASE_DIR / "stgcn_split_all",
        "label": "Raw Split",
    },
}

CLASSES      = ["Standing", "Walking", "Sitting", "Falling"]
CLASS_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def load_dataset(data_dir: Path):
    X_train = np.load(data_dir / "train_data.npy")
    X_val   = np.load(data_dir / "val_data.npy")
    with open(data_dir / "train_label.pkl", "rb") as f:
        _, y_train = pickle.load(f)
    with open(data_dir / "val_label.pkl", "rb") as f:
        _, y_val = pickle.load(f)
    X = np.concatenate([X_train, X_val], axis=0)
    y = list(y_train) + list(y_val)
    return X, y


def get_actual_lengths(data: np.ndarray):
    """Số frame thực (non-zero) của mỗi sample. data: (N, C, T, V, M)"""
    s = np.abs(data).sum(axis=(1, 3, 4))   # (N, T)
    lengths = []
    for i in range(s.shape[0]):
        nz = np.nonzero(s[i])[0]
        lengths.append(int(nz[-1]) + 1 if len(nz) > 0 else 0)
    return lengths


def collect_stats(data_dir: Path):
    X, y = load_dataset(data_dir)
    lengths = get_actual_lengths(X)
    track_counts, frame_counts = [], []
    for cls_id in range(len(CLASSES)):
        mask = [j for j, lab in enumerate(y) if lab == cls_id]
        track_counts.append(len(mask))
        frame_counts.append(sum(lengths[j] for j in mask))
    return track_counts, frame_counts


# ─── Plot function (matches reference style exactly) ─────────────────────────────
def plot_distribution(counts, total, metric, ds_label, ds_key, save_path: Path):
    """
    counts  : list of int per class
    metric  : "Track" | "Frame"
    ds_label: human-readable dataset name
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Title ───────────────────────────────────────────────────────────────────
    unit = "tracks" if metric == "Track" else "frames"
    fig.suptitle(
        f"{metric} Distribution by Action Class ({ds_label}, total = {total:,} {unit})",
        fontsize=16, fontweight="bold",
    )

    x_pos = np.arange(len(CLASSES))

    # ── Bar chart (left) ────────────────────────────────────────────────────────
    bars = ax1.bar(x_pos, counts, color=CLASS_COLORS, edgecolor="white",
                   linewidth=1.5, width=0.6)

    ax1.set_title(f"(a) {metric} Count per Class",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Action Class", fontsize=12)
    ax1.set_ylabel(f"Number of {metric}s", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(CLASSES, fontsize=11)
    ax1.set_ylim(0, max(counts) * 1.22)
    ax1.set_xlim(-0.6, len(CLASSES) - 0.4)

    # Grid (horizontal dashed, behind bars)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.7, color="#bbbbbb")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    # ── Pie chart (right) ───────────────────────────────────────────────────────
    labels_pie = [
        f"{cls}\n{cnt:,} ({cnt / total * 100:.1f}%)"
        for cls, cnt in zip(CLASSES, counts)
    ]
    ax2.pie(
        counts,
        labels=labels_pie,
        colors=CLASS_COLORS,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    ax2.set_title(f"(b) {metric} Proportion per Class",
                  fontsize=14, fontweight="bold", pad=15)
    ax2.axis("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    for ds_key, ds_cfg in DATASETS.items():
        data_dir  = ds_cfg["path"]
        ds_label  = ds_cfg["label"]

        if not data_dir.exists():
            print(f"[SKIP] Dataset not found: {data_dir}")
            continue

        print(f"\n{'='*55}")
        print(f"Dataset: {ds_label} ({ds_key})")
        track_counts, frame_counts = collect_stats(data_dir)

        total_tracks = sum(track_counts)
        total_frames = sum(frame_counts)

        for i, cls in enumerate(CLASSES):
            print(f"  {cls:10s}: {track_counts[i]:>4} tracks | "
                  f"{frame_counts[i]:>7,} frames | "
                  f"avg {frame_counts[i]/track_counts[i]:.1f} fr/track")

        # Save track distribution
        plot_distribution(
            counts    = track_counts,
            total     = total_tracks,
            metric    = "Track",
            ds_label  = ds_label,
            ds_key    = ds_key,
            save_path = BASE_DIR / f"{ds_key}_track_dist.png",
        )

        # Save frame distribution
        plot_distribution(
            counts    = frame_counts,
            total     = total_frames,
            metric    = "Frame",
            ds_label  = ds_label,
            ds_key    = ds_key,
            save_path = BASE_DIR / f"{ds_key}_frame_dist.png",
        )

    print("\nDone. Generated 4 PNG files.")


if __name__ == "__main__":
    main()
