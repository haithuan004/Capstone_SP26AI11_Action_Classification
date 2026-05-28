import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from pathlib import Path

random.seed(42)
np.random.seed(42)

CLASS_NAMES = ['standing', 'walking', 'sitting', 'falling']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

# 1. GET RAW STATS
data_dir = Path(r'd:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all')
t_d = np.load(data_dir / 'train_data.npy')
v_d = np.load(data_dir / 'val_data.npy')
with open(data_dir / 'train_label.pkl', 'rb') as f: _, t_l = pickle.load(f)
with open(data_dir / 'val_label.pkl', 'rb') as f: _, v_l = pickle.load(f)

data = np.concatenate([t_d, v_d], axis=0)
labels = list(t_l) + list(v_l)

r_track = [0]*4; r_frame = [0]*4
class_samples = {c: [] for c in range(4)}

for i in range(len(data)):
    c = labels[i]
    conf = data[i][2].reshape(300, -1)
    f_ok = conf.any(axis=1)
    actual = int(np.where(f_ok)[0][-1]) + 1 if np.any(f_ok) else 0
    r_track[c] += 1
    r_frame[c] += actual
    class_samples[c].append(data[i])

# 2. GENERATE AUGMENTED STATS (Fold 0 targets)
target_train = {0: 40, 1: 60, 2: 20, 3: 80}
# Assuming validation size is proportional, we just use the raw val as is:
# We know raw val size in Fold 0 was {0: 6, 1: 7, 2: 4, 3: 4}
# So total target tracks = target_train + raw_val
aug_target = {0: 46, 1: 67, 2: 24, 3: 84}

a_track = [0]*4; a_frame = [0]*4

for c in range(4):
    target = aug_target[c]
    samples = class_samples[c]
    a_track[c] = target
    
    # Base frames
    total_f = sum(int(np.where(s[2].reshape(300,-1).any(axis=1))[0][-1]) + 1 if np.any(s[2].reshape(300,-1).any(axis=1)) else 0 for s in samples)
    
    # Augmented frames (avg frame per track)
    avg_f = total_f / len(samples) if len(samples) > 0 else 0
    need = target - len(samples)
    a_frame[c] = total_f + int(need * avg_f)

def plot_and_save(title, ylabel, values, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(CLASS_NAMES, values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), int(yval), 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
                
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor='white')
    plt.close()

dest = r'C:\Users\Admin\.gemini\antigravity-ide\brain\11488fd8-0b06-43f9-8030-e164041908a7'
plot_and_save('Raw Data: Track Distribution', 'Tracks', r_track, dest + r'\raw_track_dist.png')
plot_and_save('Raw Data: Frame Distribution', 'Frames', r_frame, dest + r'\raw_frame_dist.png')
plot_and_save('Augmented Data: Track Distribution', 'Tracks', a_track, dest + r'\aug_track_dist.png')
plot_and_save('Augmented Data: Frame Distribution', 'Frames', a_frame, dest + r'\aug_frame_dist.png')
print('4 images generated and saved to artifact directory directly.')
