import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

JSON_PATH   = Path(r"d:\Capstone2026\Action Predict\Labeled_data\checkpoints_compare\comparison_results.json")
OUTPUT_PATH = Path(r"d:\Capstone2026\Action Predict\Labeled_data\model_comparison.png")

with open(JSON_PATH) as f:
    data = json.load(f)

models   = list(data.keys())
accuracy = [data[m]["accuracy"]  for m in models]
macro_f1 = [data[m]["macro_f1"]  for m in models]
colors   = ["#0984e3", "#d63031", "#00b894"]
labels   = ["ST-GCN", "CTR-GCN", "S-GCN"]
x = np.arange(len(models))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor="white")

for ax, vals, title in [(ax1, accuracy, "Accuracy"), (ax2, macro_f1, "Macro F1-Score")]:
    ax.set_facecolor("#f8f9fa")
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.5)

    best = int(np.argmax(vals))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#2d3436")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color="#2d3436")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylabel("Score", fontsize=11)

plt.tight_layout(pad=2.5)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {OUTPUT_PATH}")
