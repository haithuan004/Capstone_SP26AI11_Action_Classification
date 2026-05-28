import json
import matplotlib.pyplot as plt
import numpy as np

with open(r'd:\Capstone2026\Action Predict\Labeled_data\best_model_final\best_metrics_final.json', 'r') as f:
    data = json.load(f)

classes = list(data['per_class'].keys())
p_vals = [data['per_class'][c]['p'] for c in classes]
r_vals = [data['per_class'][c]['r'] for c in classes]
f1_vals = [data['per_class'][c]['f1'] for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, p_vals, width, label='Precision', color='#3498db')
rects2 = ax.bar(x, r_vals, width, label='Recall', color='#e74c3c')
rects3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='#2ecc71')

ax.set_ylabel('Scores')
ax.set_title('Performance Metrics per Class')
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in classes])
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
ax.set_ylim(0, 1.1)

ax.bar_label(rects1, fmt='%.2f', padding=3, fontsize=9)
ax.bar_label(rects2, fmt='%.2f', padding=3, fontsize=9)
ax.bar_label(rects3, fmt='%.2f', padding=3, fontsize=9)

fig.tight_layout()
plt.savefig(r'd:\Capstone2026\Action Predict\Labeled_data\best_model_final\metrics_chart.png', dpi=150)
print('Chart saved successfully.')
