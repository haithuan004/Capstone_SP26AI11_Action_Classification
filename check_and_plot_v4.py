import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def get_lengths(data):
    s = np.sum(np.abs(data), axis=(1, 3, 4))
    lengths = []
    for i in range(len(data)):
        nz = np.nonzero(s[i])[0]
        if len(nz) > 0:
            lengths.append(nz[-1] + 1)
        else:
            lengths.append(0)
    return lengths

def main():
    out_dir = Path(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_augmented_v4")
    
    X_train = np.load(out_dir / "train_data.npy")
    with open(out_dir / "train_label.pkl", "rb") as f:
        _, y_train = pickle.load(f)
        
    X_val = np.load(out_dir / "val_data.npy")
    with open(out_dir / "val_label.pkl", "rb") as f:
        _, y_val = pickle.load(f)
        
    all_data = np.concatenate([X_train, X_val], axis=0)
    all_labels = np.concatenate([y_train, y_val], axis=0)
    
    lengths = get_lengths(all_data)
    
    classes = ['Standing', 'Walking', 'Sitting', 'Falling']
    track_counts = []
    frame_counts = []
    
    for i in range(4):
        mask = (all_labels == i)
        tc = np.sum(mask)
        track_counts.append(tc)
        
        fc = sum([lengths[j] for j in range(len(lengths)) if all_labels[j] == i])
        frame_counts.append(fc)
        
    print("Final Track counts:", dict(zip(classes, track_counts)))
    print("Final Frame counts:", dict(zip(classes, frame_counts)))
    
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    total_tracks = sum(track_counts)
    total_frames = sum(frame_counts)
    
    # Plot 1: Track Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Track Distribution by Action Class (Augmented V4, total = {total_tracks:,} tracks)', 
                 fontsize=16, fontweight='bold', y=1.02)
    x_pos = np.arange(len(classes))
    bars = ax1.bar(x_pos, track_counts, color=colors, edgecolor='white', width=0.6)
    ax1.set_title('(a) Track Count per Class', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Action Class', fontsize=12)
    ax1.set_ylabel('Number of Tracks', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(0, max(track_counts) * 1.2)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for p in bars:
        ax1.annotate(f"{int(p.get_height()):,}", xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)
    
    labels_with_pct = [f"{cls}\n{count:,} ({(count/total_tracks)*100:.1f}%)" for cls, count in zip(classes, track_counts)]
    ax2.pie(track_counts, labels=labels_with_pct, colors=colors, startangle=140, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 11})
    ax2.set_title('(b) Track Proportion per Class', fontsize=14, fontweight='bold', pad=15)
    ax2.axis('equal')
    plt.tight_layout()
    plt.savefig(sys.argv[1] if len(sys.argv) > 1 else 'aug_v4_track_dist.png', bbox_inches='tight', dpi=150)

    # Plot 2: Frame Distribution
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(f'Frame Distribution by Action Class (Augmented V4, total = {total_frames:,} frames)', 
                 fontsize=16, fontweight='bold', y=1.02)
    bars2 = ax3.bar(x_pos, frame_counts, color=colors, edgecolor='white', width=0.6)
    ax3.set_title('(a) Frame Count per Class', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Action Class', fontsize=12)
    ax3.set_ylabel('Number of Frames', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes)
    ax3.set_ylim(0, max(frame_counts) * 1.2)
    ax3.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    for p in bars2:
        ax3.annotate(f"{int(p.get_height()):,}", xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)
    
    labels_with_pct2 = [f"{cls}\n{count:,} ({(count/total_frames)*100:.1f}%)" for cls, count in zip(classes, frame_counts)]
    ax4.pie(frame_counts, labels=labels_with_pct2, colors=colors, startangle=140, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 11})
    ax4.set_title('(b) Frame Proportion per Class', fontsize=14, fontweight='bold', pad=15)
    ax4.axis('equal')
    plt.tight_layout()
    plt.savefig(sys.argv[2] if len(sys.argv) > 2 else 'aug_v4_frame_dist.png', bbox_inches='tight', dpi=150)

if __name__ == "__main__":
    main()
