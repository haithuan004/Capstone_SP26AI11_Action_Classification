import numpy as np
import pickle
from pathlib import Path
from collections import Counter

data_dir = Path(r'd:\Capstone2026\Action Predict\Labeled_data\extracted_new_data')
with open(data_dir / 'train_label.pkl', 'rb') as f: _, t_l = pickle.load(f)
with open(data_dir / 'val_label.pkl', 'rb') as f: _, v_l = pickle.load(f)
l = list(t_l) + list(v_l)
print(f"Total Combined Data dist: {dict(sorted(Counter(l).items()))}")
