import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

CLASS_NAMES = ['standing', 'walking', 'sitting', 'falling']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

def get_stats(data_dir):
    data_dir = Path(data_dir)
    t_d = np.load(data_dir / 'train_data.npy')
    v_d = np.load(data_dir / 'val_data.npy')
    with open(data_dir / 'train_label.pkl', 'rb') as f: _, t_l = pickle.load(f)
    with open(data_dir / 'val_label.pkl', 'rb') as f: _, v_l = pickle.load(f)

    data = np.concatenate([t_d, v_d], axis=0)
    labels = list(t_l) + list(v_l)

    track_counts = {c: 0 for c in range(4)}
    frame_counts = {c: 0 for c in range(4)}

    for i in range(len(data)):
        c = labels[i]
        sample = data[i]
        conf = sample[2].reshape(300, -1)
        frame_ok = conf.any(axis=1)
        actual = int(np.where(frame_ok)[0][-1]) + 1 if np.any(frame_ok) else 0
        track_counts[c] += 1
        frame_counts[c] += actual
        
    return [track_counts[i] for i in range(4)], [frame_counts[i] for i in range(4)]

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

# 1. RAW DATA
raw_dir = r'd:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all'
r_tracks, r_frames = get_stats(raw_dir)
plot_and_save('Raw Data: Track Distribution', 'Tracks', r_tracks, r'd:\Capstone2026\Action Predict\Labeled_data\raw_track_dist.png')
plot_and_save('Raw Data: Frame Distribution', 'Frames', r_frames, r'd:\Capstone2026\Action Predict\Labeled_data\raw_frame_dist.png')

# 2. AUGMENTED DATA (Fold 0)
aug_dir = r'd:\Capstone2026\Action Predict\Labeled_data\New_folder_1\kfold_data\fold_0'
a_tracks, a_frames = get_stats(aug_dir)
plot_and_save('Augmented Data (Fold 0): Track Distribution', 'Tracks', a_tracks, r'd:\Capstone2026\Action Predict\Labeled_data\aug_track_dist.png')
plot_and_save('Augmented Data (Fold 0): Frame Distribution', 'Frames', a_frames, r'd:\Capstone2026\Action Predict\Labeled_data\aug_frame_dist.png')
print('Images generated successfully.')
