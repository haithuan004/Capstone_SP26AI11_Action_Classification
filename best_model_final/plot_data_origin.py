import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

CLASS_NAMES = ['Standing', 'Walking', 'Sitting', 'Falling']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'] 

def get_stats_from_zips(data_dir, exclude_file):
    zips = Path(data_dir).glob('*.zip')
    t_c = {c: 0 for c in ['standing', 'walking', 'sitting', 'falling']}
    f_c = {c: 0 for c in ['standing', 'walking', 'sitting', 'falling']}
    
    for zp in zips:
        if zp.name == exclude_file:
            continue
        try:
            with zipfile.ZipFile(zp, 'r') as zf:
                xmls = [n for n in zf.namelist() if n.endswith('.xml')]
                if not xmls: continue
                with zf.open(xmls[0]) as f:
                    root = ET.parse(f).getroot()
                    for t in root.findall('./track'):
                        acts = [a.text.lower() for sk in t.findall('./skeleton') for a in sk.findall('./attribute') if a.get('name')=='action' and a.text]
                        if not acts: continue
                        act = Counter(acts).most_common(1)[0][0]
                        if act in t_c:
                            t_c[act] += 1
                            f_c[act] += len(t.findall('./skeleton'))
        except Exception as e:
            print(f"Error reading {zp.name}: {e}")
            
    return [t_c[c.lower()] for c in CLASS_NAMES], [f_c[c.lower()] for c in CLASS_NAMES]

def make_template_chart(values, unit_name, title_prefix, out_path):
    total = sum(values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars = ax1.bar(CLASS_NAMES, values, color=COLORS, edgecolor='white', linewidth=1)
    ax1.set_title(f'(a) {unit_name} Count per Class', fontweight='bold', pad=15)
    ax1.set_ylabel(f'Number of {unit_name}s')
    ax1.set_xlabel('Action Class')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='grey')
    ax1.set_axisbelow(True)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                 int(yval), ha='center', va='bottom', color='#555555')

    if max(values) > 0:
        ax1.set_ylim(0, max(values) * 1.15)
    
    pie_labels = [f'{c}\n{v} ({(v/total)*100:.1f}%)' if total > 0 else f'{c}\n0 (0%)' for c, v in zip(CLASS_NAMES, values)]
    
    if total > 0:
        wedges, texts = ax2.pie(values, labels=pie_labels, colors=COLORS, startangle=140, 
                                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        for t in texts:
            t.set_fontsize(9)
            t.set_color('#333333')
    
    ax2.set_title(f'(b) {unit_name} Proportion per Class', fontweight='bold', pad=15)
    
    fig.suptitle(f'{title_prefix} (Data_origin_cleaned, total = {total} {unit_name.lower()}s)', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    data_dir = r'd:\Capstone2026\Action Predict\Labeled_data\Data_origin_cleaned'
    exclude_file = 'task_2260379_annotations_2026_05_21_18_59_58_cvat for images 1.1.zip'
    
    print(f'Reading XMLs from {data_dir} (excluding {exclude_file})...')
    tracks, frames = get_stats_from_zips(data_dir, exclude_file)
    print('Tracks:', tracks)
    print('Frames:', frames)
    
    out_dir = Path(r'd:\Capstone2026\Action Predict\Labeled_data\best_model_final')
    track_out = out_dir / 'track_dist_cleaned.png'
    frame_out = out_dir / 'frame_dist_cleaned.png'
    
    make_template_chart(tracks, 'Track', 'Track Distribution by Action Class', track_out)
    print(f'Saved track chart to {track_out}')
    
    make_template_chart(frames, 'Frame', 'Frame Distribution by Action Class', frame_out)
    print(f'Saved frame chart to {frame_out}')
