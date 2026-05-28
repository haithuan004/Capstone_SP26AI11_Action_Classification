import numpy as np
import pickle
from pathlib import Path
from collections import Counter

DATA_DIR = Path(r'd:\Capstone2026\Action Predict\Labeled_data\extracted_new_data')
with open(DATA_DIR / 'train_label.pkl', 'rb') as f:
    t_n, t_l = pickle.load(f)
with open(DATA_DIR / 'val_label.pkl', 'rb') as f:
    v_n, v_l = pickle.load(f)

action_plus_labels = []
for n, l in zip(t_n, t_l):
    if 'Action_plus' in n:
        action_plus_labels.append(l)
for n, l in zip(v_n, v_l):
    if 'Action_plus' in n:
        action_plus_labels.append(l)

print("Action_plus label distribution:", Counter(action_plus_labels))
