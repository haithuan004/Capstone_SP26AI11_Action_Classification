import numpy as np
import pickle

train_data = np.load(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all\train_data.npy")
val_data = np.load(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all\val_data.npy")

with open(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all\train_label.pkl", "rb") as f:
    _, train_labels = pickle.load(f)
with open(r"d:\Capstone2026\Action Predict\Labeled_data\stgcn_split_all\val_label.pkl", "rb") as f:
    _, val_labels = pickle.load(f)

all_labels = np.concatenate([train_labels, val_labels], axis=0)

from collections import Counter
print("Class counts (tracks):", Counter(all_labels))
