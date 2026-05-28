import sys
from pathlib import Path

filepath = Path(r'd:\Capstone2026\Action Predict\Labeled_data\Finetuning\finetune_fixed.py')
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'PRETRAINED = BASE_DIR / "best_model_final" / "best_stgcn.pth"',
    'PRETRAINED = BASE_DIR / "best_model_final" / "best_stgcn_final.pth"'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated PRETRAINED path")
