import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from pathlib import Path

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # nose, eyes, ears
    (5, 6),                           # shoulders
    (5, 7), (7, 9),                   # left arm
    (6, 8), (8, 10),                  # right arm
    (5, 11), (6, 12), (11, 12),       # trunk
    (11, 13), (13, 15),               # left leg
    (12, 14), (14, 16)                # right leg
]

def load_split(split_dir, prefix):
    data = np.load(split_dir / f"{prefix}_data.npy")
    with open(split_dir / f"{prefix}_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    return data, list(names), list(labels)

def plot_sequence(axes, data, title):
    # data is (C=3, T, V=17, M=1)
    C, T, V, M = data.shape
    # find valid frames
    conf = data[2].reshape(T, -1)
    frame_ok = conf.any(axis=1)
    valid_frames = np.where(frame_ok)[0]
    
    if len(valid_frames) == 0:
        return
        
    # Pick a few evenly spaced frames, say 5
    num_to_plot = len(axes)
    if len(valid_frames) >= num_to_plot:
        idx = np.linspace(0, len(valid_frames)-1, num_to_plot, dtype=int)
        frames_to_plot = valid_frames[idx]
    else:
        frames_to_plot = valid_frames
        
    for i, f in enumerate(frames_to_plot):
        if i >= num_to_plot: break
        ax = axes[i]
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
        # Plot single person M=0
        x = data[0, f, :, 0]
        y = data[1, f, :, 0]
        c = data[2, f, :, 0]
        
        for p1, p2 in SKELETON_EDGES:
            if c[p1] > 0 and c[p2] > 0:
                ax.plot([x[p1], x[p2]], [y[p1], y[p2]], color="#1f77b4", linewidth=3.5)
                
        vis = c > 0
        if np.any(vis):
            ax.scatter(x[vis], y[vis], c="#d62728", s=80, edgecolors='white', linewidths=1.5, zorder=3)
            
            pad_x = max(0.1, (max(x[vis]) - min(x[vis])) * 0.2)
            pad_y = max(0.1, (max(y[vis]) - min(y[vis])) * 0.2)
            ax.set_xlim(min(x[vis]) - pad_x, max(x[vis]) + pad_x)
            ax.set_ylim(max(y[vis]) + pad_y, min(y[vis]) - pad_y)
            
        # Add frame title below or inside the box
        ax.text(0.5, -0.1, f"Frame {f}", transform=ax.transAxes, ha='center', fontsize=10)

def main():
    aug_dir = Path("stgcn_augmented")
    orig_dir = Path("stgcn_split_all")
    
    aug_data, aug_names, aug_labels = load_split(aug_dir, "train")
    orig_data, orig_names, orig_labels = load_split(orig_dir, "train")
    
    # Find names that have been augmented (contain '_')
    aug_candidates = [i for i, name in enumerate(aug_names) if "_00" in name or "_01" in name or "_02" in name or "_03" in name or "_04" in name]
    
    if not aug_candidates:
        print("No augmented data found")
        return
        
    # Randomly pick one
    idx = random.choice(aug_candidates)
    aug_name = aug_names[idx]
    aug_seq = aug_data[idx]
    
    parts = aug_name.split('_')
    src_name = "_".join(parts[:-2])
    aug_type = parts[-2]
    
    try:
        orig_idx = orig_names.index(src_name)
        orig_seq = orig_data[orig_idx]
    except ValueError:
        print(f"Could not find original track for {src_name}")
        return
        
    print(f"Original: {src_name}")
    print(f"Augmented: {aug_name} (Type: {aug_type})")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    plot_sequence(axes[0], orig_seq, f"Original ({src_name})")
    axes[0, 2].set_title(f"Original ({src_name})", fontsize=14, pad=15, fontweight='bold')
    
    plot_sequence(axes[1], aug_seq, f"Augmented ({aug_type})")
    axes[1, 2].set_title(f"Augmented: {aug_type} ({aug_name})", fontsize=14, pad=15, fontweight='bold')
    
    plt.tight_layout()
    # Add some spacing between rows
    plt.subplots_adjust(hspace=0.4)
    
    output_path = "CapstoneProject_AIP491_FUHCM_Report_Template/visualize_augmentation.png"
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
