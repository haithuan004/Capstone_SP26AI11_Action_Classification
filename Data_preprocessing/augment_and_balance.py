"""
augment_and_balance.py
======================
Pipeline augmentation & balancing cho skeleton dataset (ST-GCN format).

Kỹ thuật augmentation:
  1. Flip-LR      : Lật ngang khung xương
  2. Rotate       : Xoay ngẫu nhiên ±15°
  3. Speed-up     : Nhanh hơn (factor 1.5x) — cắt ngắn chuỗi
  4. Slow-down    : Chậm hơn (factor 0.7x) — nội suy dài hơn
  5. Joint Noise  : Nhiễu Gaussian lên tọa độ keypoint
  6. Scale Jitter : Thay đổi tỉ lệ cơ thể ±10%
  7. Temporal Crop: Cắt ngẫu nhiên đoạn thời gian rồi resize lại

Chiến lược balancing:
  - Mục tiêu: mỗi class đạt TARGET_PER_CLASS tracks
  - Class nào ít hơn TARGET thì augment thêm cho đủ
  - Class nào đã đủ thì giữ nguyên
"""

import numpy as np
import pickle
import math
import random
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
DATA_DIR   = Path(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all")
OUT_DIR    = Path(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_augmented")

TARGET_PER_CLASS = 45    # Mục tiêu số track mỗi class sau augmentation
SEED             = 42
MAX_FRAME        = 300

CLASS_NAMES  = ["standing", "walking", "sitting", "falling"]
FLIP_PAIRS   = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────
# Augmentation functions  shape: (C=3, T, V=17, M=1)
# ──────────────────────────────────────────────────────
def flip_lr(data: np.ndarray) -> np.ndarray:
    """Lật ngang: đảo x, hoán vị các cặp khớp trái-phải."""
    out = data.copy()
    out[0] = -out[0]
    tmp = out.copy()
    for l, r in FLIP_PAIRS:
        out[:, :, l, :] = tmp[:, :, r, :]
        out[:, :, r, :] = tmp[:, :, l, :]
    return out


def rotate(data: np.ndarray, angle_deg: float = None) -> np.ndarray:
    """Xoay toàn bộ skeleton quanh trọng tâm trong mặt phẳng 2D."""
    out = data.copy()
    if angle_deg is None:
        angle_deg = random.uniform(-15, 15)
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for ti in range(out.shape[1]):
        conf = out[2, ti, :, 0]
        vis  = conf > 0
        if not np.any(vis): continue
        x = out[0, ti, :, 0].copy()
        y = out[1, ti, :, 0].copy()
        cx, cy = x[vis].mean(), y[vis].mean()
        xc = x[vis] - cx; yc = y[vis] - cy
        x[vis] = cos_t * xc - sin_t * yc + cx
        y[vis] = sin_t * xc + cos_t * yc + cy
        out[0, ti, :, 0] = x
        out[1, ti, :, 0] = y
    return out


def speed_perturb(data: np.ndarray, factor: float = None) -> np.ndarray:
    """Thay đổi tốc độ bằng cách nội suy lại theo thời gian."""
    C, T, V, M = data.shape
    if factor is None:
        factor = random.uniform(0.7, 1.4)
    # Tìm số frame thực có keypoint
    conf     = data[2].reshape(T, -1)
    frame_ok = conf.any(axis=1)
    if not np.any(frame_ok): return data.copy()
    actual = int(np.where(frame_ok)[0][-1]) + 1
    if actual < 5: return data.copy()

    new_len = int(actual / factor)
    new_len = max(5, min(new_len, T))

    out = np.zeros_like(data)
    src_idx = np.linspace(0, actual - 1, new_len)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                out[c, :new_len, v, m] = np.interp(
                    src_idx, np.arange(actual), data[c, :actual, v, m])
    out[2] = np.where(out[2] > 0.1, 1.0, 0.0)
    return out


def joint_noise(data: np.ndarray, sigma: float = None) -> np.ndarray:
    """Thêm nhiễu Gaussian lên tọa độ x, y của các keypoint hiển thị."""
    if sigma is None:
        sigma = random.uniform(0.005, 0.025)
    out = data.copy()
    mask = out[2:3] > 0   # (1, T, V, 1)
    noise = np.random.randn(2, *out.shape[1:]).astype(np.float32) * sigma
    out[0:2] += noise * mask
    return out


def scale_jitter(data: np.ndarray, scale: float = None) -> np.ndarray:
    """Thay đổi tỉ lệ skeleton ±10% (zoom in/out cơ thể)."""
    if scale is None:
        scale = random.uniform(0.90, 1.10)
    out = data.copy()
    out[0:2] *= scale
    return out


def temporal_crop(data: np.ndarray, ratio: float = None) -> np.ndarray:
    """Cắt ngẫu nhiên một đoạn (ratio*T) rồi resize lại về T frame."""
    C, T, V, M = data.shape
    if ratio is None:
        ratio = random.uniform(0.65, 0.90)
    conf     = data[2].reshape(T, -1)
    frame_ok = conf.any(axis=1)
    if not np.any(frame_ok): return data.copy()
    actual = int(np.where(frame_ok)[0][-1]) + 1
    crop_len = max(5, int(actual * ratio))
    start = random.randint(0, max(0, actual - crop_len))
    crop = data[:, start:start + crop_len]
    # Resize lại về actual length
    out = np.zeros_like(data)
    src_idx = np.linspace(0, crop_len - 1, actual)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                out[c, :actual, v, m] = np.interp(
                    src_idx, np.arange(crop_len), crop[c, :, v, m])
    out[2] = np.where(out[2] > 0.1, 1.0, 0.0)
    return out


# Danh sách các phép augmentation và tên tương ứng
AUG_REGISTRY = [
    ("flip",    flip_lr),
    ("rot+",    lambda d: rotate(d,  12)),
    ("rot-",    lambda d: rotate(d, -12)),
    ("speedup", lambda d: speed_perturb(d, 1.5)),
    ("slowdn",  lambda d: speed_perturb(d, 0.7)),
    ("noise",   joint_noise),
    ("scale+",  lambda d: scale_jitter(d, 1.10)),
    ("scale-",  lambda d: scale_jitter(d, 0.90)),
    ("tcrop",   temporal_crop),
    ("fl+rot",  lambda d: rotate(flip_lr(d), 10)),
    ("fl+ns",   lambda d: joint_noise(flip_lr(d))),
    ("fl+spd",  lambda d: speed_perturb(flip_lr(d), 1.4)),
]


def augment_to_target(samples, current_count, target, class_name):
    """Sinh thêm samples để đạt target, dùng tổ hợp augmentation ngẫu nhiên."""
    need = target - current_count
    if need <= 0:
        print(f"  [{class_name}] already {current_count} >= {target}, skip.")
        return []

    print(f"  [{class_name}] {current_count} -> {target}  (+{need} augmented)")
    new_samples = []
    aug_cycle   = list(range(len(AUG_REGISTRY)))

    for i in range(need):
        # Chọn vòng tròn qua registry để đảm bảo đa dạng
        aug_name, aug_fn = AUG_REGISTRY[i % len(AUG_REGISTRY)]
        src_name, src_data, label = random.choice(samples)
        aug_data = aug_fn(src_data)
        new_name = f"{src_name}_{aug_name}_{i:03d}"
        new_samples.append((new_name, aug_data, label))

    return new_samples


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────
def load_split(split_dir, prefix):
    data   = np.load(split_dir / f"{prefix}_data.npy")
    with open(split_dir / f"{prefix}_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    return data, list(names), list(labels)


def save_split(out_dir, prefix, data_arr, names, labels):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_data.npy", data_arr)
    with open(out_dir / f"{prefix}_label.pkl", "wb") as f:
        pickle.dump((names, labels), f)


def main():
    print("=" * 60)
    print("Augmentation & Balancing Pipeline")
    print("=" * 60)

    # Load train split (val split không augment để đánh giá khách quan)
    tr_data, tr_names, tr_labels = load_split(DATA_DIR, "train")
    va_data, va_names, va_labels = load_split(DATA_DIR, "val")

    print(f"\nTrain before: {dict(sorted(Counter(tr_labels).items()))}")
    print(f"Val   before: {dict(sorted(Counter(va_labels).items()))}")
    print(f"Train shape:  {tr_data.shape}")

    # Tách thành dict theo class
    class_samples = {c: [] for c in range(len(CLASS_NAMES))}
    for i in range(len(tr_labels)):
        cls = tr_labels[i]
        class_samples[cls].append((tr_names[i], tr_data[i], cls))

    # Augment từng class lên TARGET_PER_CLASS
    print(f"\nTarget per class: {TARGET_PER_CLASS}")
    print("-" * 40)

    all_samples = []
    for cls_id, samples in class_samples.items():
        cls_name = CLASS_NAMES[cls_id]
        new = augment_to_target(samples, len(samples), TARGET_PER_CLASS, cls_name)
        all_samples.extend(samples)
        all_samples.extend(new)

    # Shuffle
    random.shuffle(all_samples)

    # Pack
    names_out  = [s[0] for s in all_samples]
    data_out   = np.stack([s[1] for s in all_samples]).astype(np.float32)
    labels_out = [s[2] for s in all_samples]

    print(f"\nTrain after:  {dict(sorted(Counter(labels_out).items()))}")
    print(f"Train shape:  {data_out.shape}")

    # Save
    save_split(OUT_DIR, "train", data_out, names_out, labels_out)
    save_split(OUT_DIR, "val",   va_data,  va_names,  va_labels)

    print(f"\n[Saved] -> {OUT_DIR}")
    print(f"  train_data.npy  : {data_out.shape}")
    print(f"  val_data.npy    : {va_data.shape}")
    print(f"  Classes         : {CLASS_NAMES}")

    # ── Vẽ biểu đồ before/after ──────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    before = [len(class_samples[i]) for i in range(len(CLASS_NAMES))]
    after  = [Counter(labels_out)[i] for i in range(len(CLASS_NAMES))]
    colors = ["#0984e3", "#6c5ce7", "#00b894", "#d63031"]
    x = np.arange(len(CLASS_NAMES))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    b1 = ax.bar(x - w/2, before, w, label="Before", color=[c + "88" for c in colors],
                edgecolor="white", linewidth=1.5)
    b2 = ax.bar(x + w/2, after,  w, label="After",  color=colors,
                edgecolor="white", linewidth=1.5)

    for bar, v in zip(list(b1) + list(b2), before + after):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(TARGET_PER_CLASS, color="#e17055", linestyle="--", linewidth=1.5, label=f"Target ({TARGET_PER_CLASS})")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=12)
    ax.set_ylabel("Number of Tracks", fontsize=12)
    ax.set_title("Data Augmentation & Balancing — Train Set\n"
                 "Techniques: Flip, Rotate, Speed, Noise, Scale, Temporal Crop",
                 fontsize=12, fontweight="bold", color="#2d3436", pad=10)
    ax.set_ylim(0, TARGET_PER_CLASS * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, framealpha=0.8)

    out_img = Path(r"d:\Capstone2026\Action Predict\Labeled_data\augmentation_result.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n[Chart] -> {out_img}")


if __name__ == "__main__":
    main()
