"""
offline_balancing.py
====================
Offline data augmentation for class balancing.
Applies Flip-LR, Rotation, and Speed Perturbation to the 'falling' class.
"""

import numpy as np
import pickle
import math
from pathlib import Path
from collections import Counter

DATA_DIR   = Path(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all")
FALLING_IDX = 3
FLIP_PAIRS  = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]


def aug_flip_lr(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    out[0] = -out[0]
    tmp = out.copy()
    for l, r in FLIP_PAIRS:
        out[:, :, l, :] = tmp[:, :, r, :]
        out[:, :, r, :] = tmp[:, :, l, :]
    return out


def aug_rotate(data: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    out = data.copy()
    angle = (np.random.rand() * 2 - 1) * max_angle_deg
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for ti in range(out.shape[1]):
        conf = out[2, ti, :, 0]
        vis  = conf > 0
        if not np.any(vis):
            continue
        x = out[0, ti, :, 0].copy()
        y = out[1, ti, :, 0].copy()
        cx = x[vis].mean(); cy = y[vis].mean()
        xc = x[vis] - cx;   yc = y[vis] - cy
        x[vis] = cos_t * xc - sin_t * yc + cx
        y[vis] = sin_t * xc + cos_t * yc + cy
        out[0, ti, :, 0] = x
        out[1, ti, :, 0] = y
    return out


def aug_speed(data: np.ndarray, speed_factor: float = 1.5) -> np.ndarray:
    C, T, V, M = data.shape
    conf     = data[2].reshape(T, -1)
    frame_ok = conf.any(axis=1)
    if not np.any(frame_ok):
        return data.copy()
    actual_len = int(np.where(frame_ok)[0][-1]) + 1
    if actual_len < 5:
        return data.copy()
    new_len = int(actual_len / speed_factor)
    if new_len >= T: new_len = T - 1
    if new_len < 5:  new_len = 5

    out = np.zeros_like(data)
    idx = np.linspace(0, actual_len - 1, new_len)
    for c in range(C):
        for v in range(V):
            for m in range(M):
                out[c, :new_len, v, m] = np.interp(
                    idx, np.arange(actual_len), data[c, :actual_len, v, m])
    out[2] = np.where(out[2] > 0.1, 1.0, 0.0)
    return out


def balance_dataset(data, labels, names):
    print(f"Before: {dict(Counter(labels))}")
    nd, nl, nn_ = list(data), list(labels), list(names)

    for i in range(len(data)):
        if labels[i] == FALLING_IDX:
            seq = data[i]; nm = names[i]
            # Clone 1: Flip-LR
            nd.append(aug_flip_lr(seq)); nl.append(FALLING_IDX); nn_.append(nm + "_flip")
            # Clone 2: Rotate
            nd.append(aug_rotate(seq)); nl.append(FALLING_IDX); nn_.append(nm + "_rot")
            # Clone 3: Speed + Flip
            try:
                sp = aug_speed(seq, speed_factor=1.5)
                nd.append(aug_flip_lr(sp)); nl.append(FALLING_IDX); nn_.append(nm + "_spflip")
            except Exception:
                pass

    out_data = np.stack(nd)
    print(f"After:  {dict(Counter(nl))}")
    return out_data, nl, nn_


if __name__ == "__main__":
    tr_data = np.load(DATA_DIR / "train_data.npy")
    with open(DATA_DIR / "train_label.pkl", "rb") as f:
        names, labels = pickle.load(f)

    labels = list(labels); names = list(names)
    tr_bal, lab_bal, nm_bal = balance_dataset(tr_data, labels, names)

    np.save(DATA_DIR / "train_data_balanced.npy", tr_bal)
    with open(DATA_DIR / "train_label_balanced.pkl", "wb") as f:
        pickle.dump((nm_bal, lab_bal), f)

    print(f"Saved balanced dataset: {tr_bal.shape}")
