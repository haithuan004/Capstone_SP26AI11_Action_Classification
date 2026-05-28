import sys

filepath = r'd:\Capstone2026\Action Predict\Labeled_data\Finetuning\finetune_new_data.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('best_f1 = 0.0', 'best_loss = float("inf")')
content = content.replace('if vf1 > best_f1:', 'if vl < best_loss:')
content = content.replace('best_f1 = vf1', 'best_loss = vl')
content = content.replace('print(f"  *** New best F1={best_f1:.4f} saved ***")', 'print(f"  *** New best Val Loss={best_loss:.4f} (F1={vf1:.4f}) saved ***")')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Early Stopping to monitor Val Loss")
