import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

JSON_PATH   = r"d:\Capstone2026\Action Predict\Labeled_data\checkpoints\kfold_results.json"
OUTPUT_PATH = r"d:\Capstone2026\Action Predict\Labeled_data\best_metrics_table_v2.png"

CLASS_COLORS = {
    "standing": "#0984e3",
    "walking":  "#6c5ce7",
    "sitting":  "#00b894",
    "falling":  "#d63031",
}

def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Pick best fold by accuracy
    per_fold  = data["per_fold"]
    best_fold = max(per_fold, key=lambda x: x["accuracy"])
    metrics   = best_fold
    per_class = metrics["per_class"]
    classes   = list(per_class.keys())

    col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for cls in classes:
        m = per_class[cls]
        rows.append([
            cls.capitalize(),
            f"{m['precision']:.2f}",
            f"{m['recall']:.2f}",
            f"{m['f1']:.2f}",
            str(m["support"]),
        ])

    x      = np.arange(len(classes))
    width  = 0.25
    prec   = [per_class[c]["precision"] for c in classes]
    rec    = [per_class[c]["recall"]    for c in classes]
    f1     = [per_class[c]["f1"]        for c in classes]
    colors = [CLASS_COLORS.get(c, "#636e72") for c in classes]

    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(12, 4.5),
                                          gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Best Model — Accuracy: {metrics['accuracy']:.3f}  |  Macro F1: {metrics['macro_f1']:.3f}",
                 fontsize=12, fontweight="bold", color="#2d3436", y=1.02)

    ax_bar.set_facecolor("#f8f9fa")
    b1 = ax_bar.bar(x - width, prec, width, label="Precision", color=[c + "cc" for c in colors], edgecolor="white")
    b2 = ax_bar.bar(x,         rec,  width, label="Recall",    color=colors,                     edgecolor="white")
    b3 = ax_bar.bar(x + width, f1,   width, label="F1-Score",  color=[c + "88" for c in colors], edgecolor="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([c.capitalize() for c in classes], fontsize=12)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel("Score", fontsize=12)
    ax_bar.set_xlabel("Action Class", fontsize=12)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_bar.set_axisbelow(True)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_bar.legend(["Precision", "Recall", "F1-Score"], loc="upper right", fontsize=10, framealpha=0.7)

    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 2.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dfe6e9")
        if row == 0:
            cell.set_facecolor("#2d3436")
            cell.set_text_props(weight="bold", color="white")
        else:
            cls_name = classes[row - 1]
            if col == 0:
                cell.set_facecolor(CLASS_COLORS.get(cls_name, "#636e72"))
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f8f9fa" if row % 2 == 0 else "white")

    plt.tight_layout(pad=1.5)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
