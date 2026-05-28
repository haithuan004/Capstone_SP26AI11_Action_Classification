import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(r'd:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all')
CLASS_NAMES = ['standing', 'walking', 'sitting', 'falling']

# Load data
t_d = np.load(DATA_DIR / 'train_data.npy')
v_d = np.load(DATA_DIR / 'val_data.npy')
with open(DATA_DIR / 'train_label.pkl', 'rb') as f: _, t_l = pickle.load(f)
with open(DATA_DIR / 'val_label.pkl', 'rb') as f: _, v_l = pickle.load(f)

data = np.concatenate([t_d, v_d], axis=0)
labels = list(t_l) + list(v_l)

track_counts = {c: 0 for c in range(4)}
frame_counts = {c: 0 for c in range(4)}

for i in range(len(data)):
    c = labels[i]
    sample = data[i] # (3, 300, 17, 1)
    conf = sample[2].reshape(300, -1)
    frame_ok = conf.any(axis=1)
    if np.any(frame_ok):
        actual = int(np.where(frame_ok)[0][-1]) + 1
    else:
        actual = 0
    track_counts[c] += 1
    frame_counts[c] += actual

tracks = [track_counts[i] for i in range(4)]
frames = [frame_counts[i] for i in range(4)]

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Tracks
bars1 = ax1.bar(CLASS_NAMES, tracks, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_title('Number of Tracks per Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('Tracks', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot Frames
bars2 = ax2.bar(CLASS_NAMES, frames, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_title('Total Frames per Class', fontsize=14, fontweight='bold')
ax2.set_ylabel('Frames', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 100, int(yval), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
out_path = r'd:\Capstone2026\Action Predict\Labeled_data\best_model_final\data_distribution.png'
plt.savefig(out_path, dpi=200, facecolor='white')
print(f'Saved chart to {out_path}')
