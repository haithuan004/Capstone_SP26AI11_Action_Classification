import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(r'd:\Capstone2026\Action Predict\Labeled_data\new_data_split')
CLASS_NAMES = ['Standing', 'Walking', 'Sitting', 'Falling']

# Load data
t_d = np.load(DATA_DIR / 'train_data.npy')
v_d = np.load(DATA_DIR / 'val_data.npy')
with open(DATA_DIR / 'train_label.pkl', 'rb') as f: _, t_l = pickle.load(f)
with open(DATA_DIR / 'val_label.pkl', 'rb') as f: _, v_l = pickle.load(f)

labels = list(t_l) + list(v_l)

track_counts = {c: 0 for c in range(4)}

for i in range(len(labels)):
    c = labels[i]
    track_counts[c] += 1

tracks = [track_counts[i] for i in range(4)]
total_tracks = sum(tracks)

# Styling setup
plt.style.use('seaborn-v0_8-white')
colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'Track Distribution by Action Class (new_data_split, total = {total_tracks} tracks)', 
             fontsize=16, fontweight='bold', y=1.05)

# Subplot 1: Bar Chart
bars = ax1.bar(CLASS_NAMES, tracks, color=colors, edgecolor='none')
ax1.set_title('(a) Track Count per Class', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylabel('Number of Tracks', fontsize=12)
ax1.set_xlabel('Action Class', fontsize=12)
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)

# Remove top and right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add y-axis grid
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax1.set_axisbelow(True)

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(tracks)*0.02), 
             int(yval), ha='center', va='bottom', color='#555555', fontsize=11)

# Subplot 2: Pie Chart
ax2.set_title('(b) Track Proportion per Class', fontsize=14, fontweight='bold', pad=20)

pie_labels = [f"{cls}\n{val} ({val/total_tracks*100:.1f}%)" for cls, val in zip(CLASS_NAMES, tracks)]

wedges, texts = ax2.pie(tracks, labels=pie_labels, colors=colors, 
                        startangle=140, 
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                        textprops={'fontsize': 10, 'color': '#333333'})

# Adjust layout
plt.tight_layout()
out_path = r'd:\Capstone2026\Action Predict\Labeled_data\new_data_track_distribution_styled.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved chart to {out_path}')
