import numpy as np
import pickle
import math
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold

BASE_DIR   = Path(r"d:\Capstone2026\Action Predict\Labeled_data")
DATA_DIR   = BASE_DIR / "extracted_new_data"
KFOLD_DIR  = BASE_DIR / "kfold_new_data"

# Slightly higher targets to account for new data
TARGET_PER_CLASS = {0: 60, 1: 80, 2: 40, 3: 100}
SEED             = 42
CLASS_NAMES      = ["standing", "walking", "sitting", "falling"]
FLIP_PAIRS       = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

random.seed(SEED)
np.random.seed(SEED)

def flip_lr(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    out[0] = -out[0]
    tmp = out.copy()
    for l, r in FLIP_PAIRS:
        out[:, :, l, :] = tmp[:, :, r, :]
        out[:, :, r, :] = tmp[:, :, l, :]
    return out

def rotate(data: np.ndarray, angle_deg: float = None) -> np.ndarray:
    out = data.copy()
    if angle_deg is None: angle_deg = random.uniform(-15, 15)
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
    C, T, V, M = data.shape
    if factor is None: factor = random.uniform(0.7, 1.4)
    conf = data[2].reshape(T, -1)
    frame_ok = conf.any(axis=1)
    if not np.any(frame_ok): return data.copy()
    actual = int(np.where(frame_ok)[0][-1]) + 1
    if actual < 5: return data.copy()
    new_len = max(5, min(int(actual / factor), T))
    out = np.zeros_like(data)
    src_idx = np.linspace(0, actual - 1, new_len)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                out[c, :new_len, v, m] = np.interp(src_idx, np.arange(actual), data[c, :actual, v, m])
    out[2] = np.where(out[2] > 0.1, 1.0, 0.0)
    return out

def joint_noise(data: np.ndarray, sigma: float = None) -> np.ndarray:
    if sigma is None: sigma = random.uniform(0.005, 0.025)
    out = data.copy()
    mask = out[2:3] > 0
    noise = np.random.randn(2, *out.shape[1:]).astype(np.float32) * sigma
    out[0:2] += noise * mask
    return out

def scale_jitter(data: np.ndarray, scale: float = None) -> np.ndarray:
    if scale is None: scale = random.uniform(0.90, 1.10)
    out = data.copy()
    out[0:2] *= scale
    return out

def temporal_crop(data: np.ndarray, ratio: float = None) -> np.ndarray:
    C, T, V, M = data.shape
    if ratio is None: ratio = random.uniform(0.65, 0.90)
    conf = data[2].reshape(T, -1)
    frame_ok = conf.any(axis=1)
    if not np.any(frame_ok): return data.copy()
    actual = int(np.where(frame_ok)[0][-1]) + 1
    crop_len = max(5, int(actual * ratio))
    start = random.randint(0, max(0, actual - crop_len))
    crop = data[:, start:start + crop_len]
    out = np.zeros_like(data)
    src_idx = np.linspace(0, crop_len - 1, actual)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                out[c, :actual, v, m] = np.interp(src_idx, np.arange(crop_len), crop[c, :, v, m])
    out[2] = np.where(out[2] > 0.1, 1.0, 0.0)
    return out

def joint_masking(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    num_mask = random.randint(1, 3)
    V = data.shape[2]
    mask_idx = random.sample(range(V), num_mask)
    for v in mask_idx:
        out[:, :, v, :] = 0.0
    return out

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
    ("mask",    joint_masking),
    ("fl+rot",  lambda d: rotate(flip_lr(d), 10)),
    ("fl+msk",  lambda d: joint_masking(flip_lr(d))),
]

def augment_to_target(samples, target, class_name):
    current_count = len(samples)
    need = target - current_count
    if need <= 0: return []
    new_samples = []
    for i in range(need):
        aug_name, aug_fn = AUG_REGISTRY[i % len(AUG_REGISTRY)]
        src_name, src_data, label = random.choice(samples)
        aug_data = aug_fn(src_data)
        new_name = f"{src_name}_{aug_name}_{i:03d}"
        new_samples.append((new_name, aug_data, label))
    return new_samples

def load_extracted(split_dir):
    t_d = np.load(split_dir / "train_data.npy")
    v_d = np.load(split_dir / "val_data.npy")
    with open(split_dir / "train_label.pkl", "rb") as f: t_n, t_l = pickle.load(f)
    with open(split_dir / "val_label.pkl", "rb") as f: v_n, v_l = pickle.load(f)
    data = np.concatenate([t_d, v_d], axis=0)
    names = list(t_n) + list(v_n)
    labels = list(t_l) + list(v_l)
    return data, np.array(names), np.array(labels)

def main():
    data, names, labels = load_extracted(DATA_DIR)
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n--- FOLD {fold} ---")
        fold_dir = KFOLD_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        tr_d, tr_n, tr_l = data[train_idx], names[train_idx], labels[train_idx]
        va_d, va_n, va_l = data[val_idx], names[val_idx], labels[val_idx]
        class_samples = {c: [] for c in range(len(CLASS_NAMES))}
        for i in range(len(tr_l)):
            class_samples[tr_l[i]].append((tr_n[i], tr_d[i], tr_l[i]))
        all_tr_samples = []
        for cls_id, samples in class_samples.items():
            new_samples = augment_to_target(samples, TARGET_PER_CLASS[cls_id], CLASS_NAMES[cls_id])
            all_tr_samples.extend(samples)
            all_tr_samples.extend(new_samples)
        random.shuffle(all_tr_samples)
        tr_out_names = [s[0] for s in all_tr_samples]
        tr_out_data = np.stack([s[1] for s in all_tr_samples]).astype(np.float32)
        tr_out_labels = [s[2] for s in all_tr_samples]
        print(f"Aug Train Dist: {dict(sorted(Counter(tr_out_labels).items()))}")
        print(f"Val Dist: {dict(sorted(Counter(va_l).items()))}")
        np.save(fold_dir / "train_data.npy", tr_out_data)
        with open(fold_dir / "train_label.pkl", "wb") as f: pickle.dump((tr_out_names, tr_out_labels), f)
        np.save(fold_dir / "val_data.npy", va_d)
        with open(fold_dir / "val_label.pkl", "wb") as f: pickle.dump((va_n, va_l), f)

if __name__ == "__main__":
    main()
