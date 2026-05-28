import numpy as np
import pickle
from pathlib import Path

def print_dist(path):
    p = Path(path)
    if not (p / 'train_data.npy').exists():
        print(f"Not found: {p}")
        return
    with open(p / 'train_label.pkl', 'rb') as f: _, l = pickle.load(f)
    from collections import Counter
    print(f"{p.name}: {dict(sorted(Counter(l).items()))}")

print_dist(r'd:\Capstone2026\Action Predict\Labeled_data\stgcn_augmented_v5')
print_dist(r'd:\Capstone2026\Action Predict\Labeled_data\stgcn_augmented')
