"""
generate_comparison_chart.py
============================
Reads the best metrics JSON files for ST-GCN, CTR-GCN (AGCN), and S-GCN 
and generates a bar chart comparing their Accuracy and Macro F1-Score.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR    = Path(r"d:\Capstone2026\Action Predict\Labeled_data\checkpoints_v4")
OUTPUT_PATH = Path(r"d:\Capstone2026\Action Predict\Labeled_data\model_comparison_v4.png")

# Map display names to their corresponding JSON file names
MODELS = {
    "ST-GCN":  "best_metrics_augmented.json",
    "CTR-GCN": "best_metrics_agcn_augmented.json",
    "S-GCN":   "best_metrics_sgcn_augmented.json"
}

def main():
    accuracy = []
    macro_f1 = []
    valid_models = []

    for name, filename in MODELS.items():
        filepath = SAVE_DIR / filename
        if not filepath.exists():
            print(f"[WARN] Metric file not found for {name} ({filepath}). Skipping...")
            continue
            
        with open(filepath, "r") as f:
            data = json.load(f)
            
        accuracy.append(data.get("accuracy", 0.0))
        macro_f1.append(data.get("macro_f1", 0.0))
        valid_models.append(name)

    if not valid_models:
        print("[ERROR] No metric files found. Please run the finetune scripts first.")
        return

    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────────────
    colors   = ["#0984e3", "#d63031", "#00b894"]  # Blue for ST, Red for CTR, Green for S
    x = np.arange(len(valid_models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor="white")

    for ax, vals, title in [(ax1, accuracy, "Accuracy"), (ax2, macro_f1, "Macro F1-Score")]:
        ax.set_facecolor("#f8f9fa")
        
        # Draw bars (match color to model index)
        bar_colors = [colors[list(MODELS.keys()).index(m)] for m in valid_models]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)

        # Highlight best model
        best = int(np.argmax(vals))
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)

        # Add text labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color="#2d3436")

        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_models, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#2d3436")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Score", fontsize=11)

    plt.suptitle("Model Performance Comparison (Augmented Data)", 
                 fontsize=15, fontweight="bold", color="#2d3436", y=1.05)
    plt.tight_layout(pad=2.0)
    
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    
    print(f"\n[OK] Generated comparison chart: {OUTPUT_PATH}")
    print("Scores:")
    for i, m in enumerate(valid_models):
        print(f"  - {m:8s}: Accuracy = {accuracy[i]:.3f}, Macro F1 = {macro_f1[i]:.3f}")

if __name__ == "__main__":
    main()
