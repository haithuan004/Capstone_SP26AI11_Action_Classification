import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
import random
import scipy.interpolate

# Constants
CLASS_TO_ID = {"standing": 0, "walking": 1, "sitting": 2, "falling": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
LABEL_TO_IDX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
MAX_T = 300

# Parsing functions
def collect_action_from_skeleton(sk: ET.Element) -> str | None:
    for attr in sk.findall("./attribute"):
        if attr.get("name") == "action" and attr.text:
            a = attr.text.strip().lower()
            if a in CLASS_TO_ID: return a
    return None

def parse_xy(pts_str: str | None):
    if not pts_str: return None
    parts = pts_str.split(",")
    if len(parts) < 2: return None
    try: return float(parts[0]), float(parts[1])
    except ValueError: return None

def skeleton_to_xy_conf(sk: ET.Element) -> np.ndarray:
    out = np.zeros((17, 3), dtype=np.float32)
    for pt in sk.findall("./points"):
        lab = pt.get("label")
        if lab not in LABEL_TO_IDX: continue
        idx = LABEL_TO_IDX[lab]
        outside = pt.get("outside") == "1"
        xy = parse_xy(pt.get("points"))
        if xy is None or outside:
            out[idx] = [0.0, 0.0, 0.0]
        else:
            out[idx] = [xy[0], xy[1], 1.0]
    return out

def majority_label_from_track(track: ET.Element) -> int | None:
    actions = []
    for sk in track.findall("./skeleton"):
        a = collect_action_from_skeleton(sk)
        if a is not None: actions.append(CLASS_TO_ID[a])
    if not actions: return None
    return Counter(actions).most_common(1)[0][0]

def normalize_sequence_inplace(data: np.ndarray) -> None:
    _, t, _, _ = data.shape
    ls, rs, lh, rh = 5, 6, 11, 12
    for ti in range(t):
        conf = data[2, ti, :, 0]
        xy = data[0:2, ti, :, 0].copy()

        shoulder_w = np.linalg.norm(xy[:, ls] - xy[:, rs]) if conf[ls] > 0 and conf[rs] > 0 else 0.0
        hip_w = np.linalg.norm(xy[:, lh] - xy[:, rh]) if conf[lh] > 0 and conf[rh] > 0 else 0.0
        torso = 0.0
        if conf[ls] > 0 and conf[rs] > 0 and conf[lh] > 0 and conf[rh] > 0:
            sm = (xy[:, ls] + xy[:, rs]) * 0.5
            hm = (xy[:, lh] + xy[:, rh]) * 0.5
            torso = float(np.linalg.norm(sm - hm))
        scale = max(shoulder_w, hip_w, torso, 1e-3)

        if conf[lh] > 0 and conf[rh] > 0: c = (xy[:, lh] + xy[:, rh]) * 0.5
        elif conf[ls] > 0 and conf[rs] > 0: c = (xy[:, ls] + xy[:, rs]) * 0.5
        elif conf[0] > 0: c = xy[:, 0]
        else:
            vis = conf > 0
            if not np.any(vis): continue
            c = xy[:, vis].mean(axis=1)

        data[0:2, ti, :, 0] = (xy - c.reshape(2, 1)) / scale

def extract_track(track: ET.Element):
    frame_map = {}
    for sk in track.findall("./skeleton"):
        ft = sk.get("frame")
        if ft is None or not str(ft).isdigit(): continue
        frame_map[int(ft)] = sk
    if not frame_map: return None, None
    label = majority_label_from_track(track)
    if label is None: return None, None
    frames = sorted(frame_map.keys())
    raw = np.zeros((3, len(frames), 17, 1), dtype=np.float32)
    for ti, f in enumerate(frames):
        xyv = skeleton_to_xy_conf(frame_map[f])
        raw[0, ti, :, 0] = xyv[:, 0]
        raw[1, ti, :, 0] = xyv[:, 1]
        raw[2, ti, :, 0] = xyv[:, 2]
    normalize_sequence_inplace(raw)
    return raw, label

def extract_image_list(root: ET.Element):
    images = root.findall("./image")
    if not images: return None, None
    def sort_key(img: ET.Element):
        i = img.get("id") or img.get("name") or "0"
        try: return int(str(i).split(".")[0])
        except: return 0
    images = sorted(images, key=sort_key)
    actions = []
    raw = np.zeros((3, len(images), 17, 1), dtype=np.float32)
    for ti, img in enumerate(images):
        sks = img.findall("./skeleton")
        if not sks: continue
        xyv = skeleton_to_xy_conf(sks[0])
        raw[0, ti, :, 0] = xyv[:, 0]
        raw[1, ti, :, 0] = xyv[:, 1]
        raw[2, ti, :, 0] = xyv[:, 2]
        a = collect_action_from_skeleton(sks[0])
        if a is not None: actions.append(CLASS_TO_ID[a])
    if not np.any(raw[2] > 0): return None, None
    label = Counter(actions).most_common(1)[0][0] if actions else None
    if label is None: return None, None
    normalize_sequence_inplace(raw)
    return raw, label

# Augmentation functions
def aug_flip(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    out[0] = -out[0]
    swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    for (l, r) in swap_pairs:
        temp = out[:, :, l, :].copy()
        out[:, :, l, :] = out[:, :, r, :]
        out[:, :, r, :] = temp
    return out

def aug_noise(data: np.ndarray, std=0.02) -> np.ndarray:
    out = data.copy()
    noise = np.random.normal(0, std, size=out[0:2].shape).astype(np.float32)
    mask = (out[2] > 0)[np.newaxis, ...]
    out[0:2] += noise * mask
    return out

def aug_occlusion(data: np.ndarray, num_joints=2) -> np.ndarray:
    out = data.copy()
    joints = np.random.choice(17, num_joints, replace=False)
    for j in joints:
        # Zero out x, y, conf
        out[:, :, j, :] = 0.0
    return out

def aug_temporal_scale(data: np.ndarray, scale: float) -> np.ndarray:
    C, T, V, M = data.shape
    new_T = int(T * scale)
    if new_T < 10: return data.copy() # Too short
    out = np.zeros((C, new_T, V, M), dtype=np.float32)
    x = np.arange(T)
    x_new = np.linspace(0, T - 1, new_T)
    for c in range(2): # Only interpolate x and y
        for v in range(V):
            f = scipy.interpolate.interp1d(x, data[c, :, v, 0], kind='linear', fill_value="extrapolate")
            out[c, :, v, 0] = f(x_new)
    # Confidence is nearest neighbor
    f_conf = scipy.interpolate.interp1d(x, data[2, :, :, 0], kind='nearest', axis=0, fill_value="extrapolate")
    out[2, :, :, 0] = f_conf(x_new)
    return out

def pad_to_max(seq: np.ndarray, max_t: int = MAX_T) -> np.ndarray:
    c, t, v, m = seq.shape
    if t == max_t: return seq
    out = np.zeros((c, max_t, v, m), dtype=np.float32)
    # Truncate if too long
    truncate_t = min(t, max_t)
    out[:, :truncate_t, :, :] = seq[:, :truncate_t, :, :]
    return out

def main():
    data_dir = Path(r"d:\Capstone2026\Action Predict\Labeled_data\Data_origin")
    zips = list(data_dir.glob("*.zip"))
    all_tracks = []
    
    print("Reading raw tracks...")
    for zp in zips:
        if zp.name.endswith("_backup.zip"): continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
                if not xml_name: continue
                with zf.open(xml_name) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    tracks = root.findall("./track")
                    if tracks:
                        for tr in tracks:
                            arr, lbl = extract_track(tr)
                            if arr is not None and lbl is not None:
                                all_tracks.append((arr, lbl))
                    else:
                        arr, lbl = extract_image_list(root)
                        if arr is not None and lbl is not None:
                            all_tracks.append((arr, lbl))
        except Exception as e:
            print(f"Error parsing {zp.name}: {e}")

    print(f"Extracted {len(all_tracks)} raw tracks.")
    
    # Train/Val Split (80/20) - Stratified
    X = [t[0] for t in all_tracks]
    y = [t[1] for t in all_tracks]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Split: {len(X_train)} Train, {len(X_val)} Val")
    
    # Chunking on Train
    print("Chunking training tracks > 100 frames (except falling)...")
    chunked_train = []
    for arr, lbl in zip(X_train, y_train):
        length = arr.shape[1]
        class_name = ID_TO_CLASS[lbl]
        if class_name != "falling" and length > 100:
            for i in range(0, length, 50):
                chunked_train.append((arr[:, i:i+50, :, :], lbl))
        else:
            chunked_train.append((arr, lbl))
            
    # Count current
    current_counts = Counter([t[1] for t in chunked_train])
    print(f"Train counts after chunking: {{ID_TO_CLASS[k]: v for k, v in current_counts.items()}}")
    
    # Target counts
    targets = {
        CLASS_TO_ID["falling"]: 130,
        CLASS_TO_ID["standing"]: 100,
        CLASS_TO_ID["walking"]: 100,
        CLASS_TO_ID["sitting"]: 100
    }
    
    # Augmentation
    print("Applying offline augmentations to reach targets...")
    augmented_train = []
    
    for class_id, target in targets.items():
        class_tracks = [t for t in chunked_train if t[1] == class_id]
        if not class_tracks: continue
        current_num = len(class_tracks)
        
        # Strategy generator
        def get_aug_variant(idx, arr):
            if idx == 0: return arr
            elif idx == 1: return aug_flip(arr)
            elif idx == 2: return aug_noise(arr)
            elif idx == 3: return aug_temporal_scale(arr, 1.2) # Fast
            elif idx == 4: return aug_temporal_scale(arr, 0.8) # Slow
            elif idx == 5: return aug_occlusion(arr)
            elif idx == 6: return aug_noise(aug_flip(arr))
            elif idx == 7: return aug_temporal_scale(aug_flip(arr), 1.2)
            elif idx == 8: return aug_occlusion(aug_flip(arr))
            else:
                # Random combination
                out = arr.copy()
                if random.random() > 0.5: out = aug_flip(out)
                if random.random() > 0.5: out = aug_noise(out)
                if random.random() > 0.5: out = aug_occlusion(out)
                return out

        multiplier = target // current_num
        remainder = target % current_num
        
        # Multiply fully
        for track_idx, (arr, lbl) in enumerate(class_tracks):
            for variant_idx in range(multiplier):
                augmented_train.append((get_aug_variant(variant_idx, arr), lbl))
                
        # Remainder
        for track_idx in range(remainder):
            arr, lbl = class_tracks[track_idx]
            augmented_train.append((get_aug_variant(multiplier, arr), lbl))
            
    final_train_counts = Counter([t[1] for t in augmented_train])
    print(f"Final Train counts: {{ID_TO_CLASS[k]: v for k, v in final_train_counts.items()}}")
    
    # Pad and format Train
    print("Padding and formatting Train data...")
    out_X_train = np.stack([pad_to_max(t[0]) for t in augmented_train])
    out_y_train = np.array([t[1] for t in augmented_train])
    
    # Pad and format Val
    print("Padding and formatting Val data...")
    out_X_val = np.stack([pad_to_max(arr) for arr in X_val])
    out_y_val = np.array(y_val)
    
    # Save
    out_dir = Path(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_augmented_v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "train_data.npy", out_X_train)
    with open(out_dir / "train_label.pkl", "wb") as f:
        pickle.dump(("Train", out_y_train), f)
        
    np.save(out_dir / "val_data.npy", out_X_val)
    with open(out_dir / "val_label.pkl", "wb") as f:
        pickle.dump(("Val", out_y_val), f)
        
    print(f"Saved successfully to {out_dir}")

if __name__ == "__main__":
    main()
