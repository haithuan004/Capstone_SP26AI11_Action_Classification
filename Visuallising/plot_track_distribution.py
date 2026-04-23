"""
plot_track_distribution.py
Đọc tất cả zip trong Data_origin, đếm số track theo class,
vẽ biểu đồ phân phối không phân biệt file zip.
"""

import zipfile, xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ZIP_DIR    = Path(r"d:\Capstone2026\Action Predict\Labeled_data\Data_origin")
OUTPUT     = Path(r"d:\Capstone2026\Action Predict\Labeled_data\track_distribution.png")

ACTION_CLASSES = ["standing", "walking", "sitting", "falling"]
CLASS_COLORS   = {
    "standing": "#0984e3",
    "walking":  "#6c5ce7",
    "sitting":  "#00b894",
    "falling":  "#d63031",
}

def get_majority_action(track_el):
    """Trả về action label phổ biến nhất trong một track."""
    counts = Counter()
    # Track-based format (video)
    for sk in track_el.findall("./skeleton"):
        for attr in sk.findall("./attribute"):
            if attr.get("name") == "action" and attr.text:
                a = attr.text.strip().lower()
                if a in ACTION_CLASSES:
                    counts[a] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return None

def get_action_from_image_skeleton(sk_el):
    for attr in sk_el.findall("./attribute"):
        if attr.get("name") == "action" and attr.text:
            a = attr.text.strip().lower()
            if a in ACTION_CLASSES:
                return a
    return None

def read_xml_from_zip(zpath):
    with zipfile.ZipFile(zpath, "r") as zf:
        xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
        if not xml_name:
            return None
        with zf.open(xml_name) as f:
            return ET.parse(f).getroot()

def count_tracks_in_zip(zpath):
    """Return Counter {class: num_tracks}"""
    try:
        root = read_xml_from_zip(zpath)
    except Exception as e:
        print(f"  [WARN] {zpath.name}: {e}")
        return Counter()

    if root is None:
        return Counter()

    counts = Counter()

    # Format 1: <track> elements (video annotation)
    tracks = root.findall("./track")
    if tracks:
        for track in tracks:
            action = get_majority_action(track)
            if action:
                counts[action] += 1
        return counts

    # Format 2: <image> elements (frame-by-frame)
    images = root.findall("./image")
    seen_actions = Counter()
    for img in images:
        for sk in img.findall("./skeleton"):
            a = get_action_from_image_skeleton(sk)
            if a:
                seen_actions[a] += 1
    # Treat whole file as 1 track per dominant action
    if seen_actions:
        counts[seen_actions.most_common(1)[0][0]] += 1

    return counts

def main():
    zips = sorted(ZIP_DIR.glob("*.zip"))
    if not zips:
        print(f"Không tìm thấy zip nào trong: {ZIP_DIR}")
        return

    total = Counter()
    print(f"Reading {len(zips)} zip files...\n")
    for zp in zips:
        c = count_tracks_in_zip(zp)
        print(f"  {zp.name:35s}  ->  {dict(c)}")
        total += c

    print(f"\nTotal: {dict(total)}")

    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────────────
    classes = ACTION_CLASSES
    values  = [total.get(c, 0) for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=1.5, width=0.55)

    # Nhãn giá trị trên đầu mỗi cột
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                str(v), ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#2d3436")

    ax.set_xlabel("Action Class", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Tracks", fontsize=12, labelpad=8)
    ax.set_title(f"Track Distribution by Action Class\n"
                 f"Total: {sum(values)} tracks  |  {len(zips)} zip files",
                 fontsize=13, fontweight="bold", color="#2d3436", pad=12)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=12)
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Phần trăm trên mỗi cột
    total_sum = sum(values)
    for bar, v in zip(bars, values):
        pct = v / total_sum * 100 if total_sum > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(values) * 0.08,
                f"({pct:.1f}%)", ha="center", va="bottom",
                fontsize=10, color="#636e72")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved -> {OUTPUT}")

if __name__ == "__main__":
    main()
