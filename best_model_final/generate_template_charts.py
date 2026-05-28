import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

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

CLASS_NAMES = ['Standing', 'Walking', 'Sitting', 'Falling']
# Colors similar to the image template
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'] 

def make_template_chart(values, unit_name, title_prefix, out_path):
    total = sum(values)
    
    # Create figure with 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Bar Chart
    bars = ax1.bar(CLASS_NAMES, values, color=COLORS, edgecolor='white', linewidth=1)
    ax1.set_title(f'(a) {unit_name} Count per Class', fontweight='bold', pad=15)
    ax1.set_ylabel(f'Number of {unit_name}s')
    ax1.set_xlabel('Action Class')
    
    # Add horizontal grid lines
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='grey')
    ax1.set_axisbelow(True) # Put grid behind bars
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                 int(yval), ha='center', va='bottom', color='#555555')

    # Extend Y-axis max a bit for labels
    ax1.set_ylim(0, max(values) * 1.15)
    
    # Subplot 2: Pie Chart
    # Format labels for pie chart: ClassName\nValue (Percentage%)
    pie_labels = [f'{c}\n{v} ({(v/total)*100:.1f}%)' for c, v in zip(CLASS_NAMES, values)]
    
    wedges, texts = ax2.pie(values, labels=pie_labels, colors=COLORS, startangle=140, 
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    
    # Pie text formatting
    for t in texts:
        t.set_fontsize(9)
        t.set_color('#333333')
        
    ax2.set_title(f'(b) {unit_name} Proportion per Class', fontweight='bold', pad=15)
    
    # Global Title
    fig.suptitle(f'{title_prefix} (Train + Val, total = {total} {unit_name.lower()}s)', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    raw_dir = r'd:\Capstone2026\Action Predict\Labeled_data\stgcn_raw_data'
    print(f'Loading data from {raw_dir}...')
    tracks, frames = get_stats(raw_dir)
    
    out_dir = Path(r'd:\Capstone2026\Action Predict\Labeled_data\best_model_final')
    track_out = out_dir / 'track_distribution_template.png'
    frame_out = out_dir / 'frame_distribution_template.png'
    
    make_template_chart(tracks, 'Track', 'Track Distribution by Action Class', track_out)
    print(f'Saved track chart to {track_out}')
    
    make_template_chart(frames, 'Frame', 'Frame Distribution by Action Class', frame_out)
    print(f'Saved frame chart to {frame_out}')
