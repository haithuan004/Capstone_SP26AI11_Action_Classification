import sys
from pathlib import Path

filepath = Path(r'd:\Capstone2026\Action Predict\Labeled_data\Finetuning\finetune_new_data.py')
outpath = Path(r'd:\Capstone2026\Action Predict\Labeled_data\Finetuning\finetune_fixed.py')

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace PRETRAINED
content = content.replace(
    'PRETRAINED = BASE_DIR / "stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth"',
    'PRETRAINED = BASE_DIR / "best_model_final" / "best_stgcn.pth"'
)

# Replace SAVE_DIR
content = content.replace(
    'SAVE_DIR   = BASE_DIR / "new_best"',
    'SAVE_DIR   = BASE_DIR / "fixed_best"'
)

# Update class_weights to demote standing (index 0)
new_class_weights = '''def class_weights(labels, nc, boost_falling=1.5):
    c = Counter(labels); total = len(labels)
    w = torch.zeros(nc)
    for i in range(nc):
        w[i] = total / (nc * max(c.get(i,1), 1))
    w[FALLING_IDX] *= boost_falling
    # Giảm trọng số của Standing (class 0) xuống 0.5 để bớt bias
    w[0] *= 0.5
    return w / w.sum() * nc'''

old_class_weights = '''def class_weights(labels, nc, boost_falling=1.5):
    c = Counter(labels); total = len(labels)
    w = torch.zeros(nc)
    for i in range(nc):
        w[i] = total / (nc * max(c.get(i,1), 1))
    w[FALLING_IDX] *= boost_falling
    return w / w.sum() * nc'''

content = content.replace(old_class_weights, new_class_weights)

# Update Focal Loss config in main
content = content.replace(
    'criterion = FocalLoss(weight=cw, gamma=2.0, ls=0.2)',
    'criterion = FocalLoss(weight=cw, gamma=3.0, ls=0.4)'
)

# Update wandb project
content = content.replace(
    'p.add_argument("--wandb-project",type=str,   default="stgcn-new-data")',
    'p.add_argument("--wandb-project",type=str,   default="stgcn-fixed-bias")'
)

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Created finetune_fixed.py")
