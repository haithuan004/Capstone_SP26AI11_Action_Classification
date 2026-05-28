import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path(r'd:\Capstone2026\Action Predict\Labeled_data\extracted_new_data')
with open(DATA_DIR / 'train_label.pkl', 'rb') as f:
    names, _ = pickle.load(f)
for n in names:
    if 'Action_plus' in n:
        print("Found:", n)
        break
else:
    print("Action_plus not found in extracted_new_data")
