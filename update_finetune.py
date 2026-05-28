import sys

filepath = r'd:\Capstone2026\Action Predict\Labeled_data\Finetuning\finetune_new_data.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('DATA_DIR   = BASE_DIR / "new_data_split"', 'DATA_DIR   = BASE_DIR / "kfold_new_data" / "fold_0"')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated finetune_new_data.py DATA_DIR")
