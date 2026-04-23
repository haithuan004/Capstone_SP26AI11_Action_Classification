import copy
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt

# Import the preprocessing functions from the provided scripts
from interpolate_keypoints_in_zips import interpolate_track, read_xml_from_zip
from kinematic_spatial_impute_zips import impute_one_skeleton
from pipeline_calibration_temporal_kinematic import (
    calibrate_track_limb_lengths,
    apply_limb_length_clamp,
    ORDERED_LIMB_SEGMENTS
)

SKELETON_EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
    ("right_eye", "right_ear"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]

def parse_xy(value):
    if not value: return None
    parts = value.split(',')
    if len(parts) < 2: return None
    try: return float(parts[0]), float(parts[1])
    except: return None

def skeleton_points_map(skeleton_node):
    points_map = {}
    for point_node in skeleton_node.findall("./points"):
        label = point_node.get("label")
        xy = parse_xy(point_node.get("points"))
        outside = point_node.get("outside")
        if not label or xy is None: continue
        if outside == "1": continue
        points_map[label] = xy
    return points_map

import random

def find_candidate_track_and_frame(data_dir: Path):
    zip_files = list(data_dir.glob("*.zip"))
    random.shuffle(zip_files)
    
    for zip_path in zip_files:
        try:
            _, root = read_xml_from_zip(zip_path)
        except Exception:
            continue
            
        tracks = root.findall("./track")
        random.shuffle(tracks)
        for track in tracks:
            skeletons = track.findall("./skeleton")
            # We want a track with at least a few frames
            if len(skeletons) < 10: continue
            
            # Find all frames with missing keypoints
            candidate_frames = []
            for sk in skeletons:
                pts = skeleton_points_map(sk)
                if 5 < len(pts) < 17:
                    candidate_frames.append(int(sk.get("frame")))
                    
            if candidate_frames:
                frame_id = random.choice(candidate_frames)
                return zip_path.name, copy.deepcopy(track), frame_id
                    
    return None, None, None

def plot_skeleton(ax, points_map, title):
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    # Draw bones
    for p1, p2 in SKELETON_EDGES:
        if p1 in points_map and p2 in points_map:
            x1, y1 = points_map[p1]
            x2, y2 = points_map[p2]
            ax.plot([x1, x2], [y1, y2], color="#1f77b4", linewidth=3.5)
            
    # Draw joints
    xs = [xy[0] for xy in points_map.values()]
    ys = [xy[1] for xy in points_map.values()]
    if xs and ys:
        ax.scatter(xs, ys, c="#d62728", s=70)
        
        pad_x = max(8.0, (max(xs) - min(xs)) * 0.15)
        pad_y = max(8.0, (max(ys) - min(ys)) * 0.15)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(max(ys) + pad_y, min(ys) - pad_y)
        
    ax.set_title(title, fontsize=10)

def main():
    data_dir = Path("Data_origin")
    zip_name, track, frame_id = find_candidate_track_and_frame(data_dir)
    
    if track is None:
        print("Could not find a suitable candidate track.")
        return
        
    print(f"Found candidate in {zip_name}, frame {frame_id}")
    
    # Extract original skeleton at F
    sk_original = None
    for sk in track.findall("./skeleton"):
        if int(sk.get("frame")) == frame_id:
            sk_original = copy.deepcopy(sk)
            break
            
    # Step 1: Temporal Interpolation
    # We interpolate the track first
    track_interp = copy.deepcopy(track)
    interpolate_track(track_interp)
    
    sk_temporal = None
    for sk in track_interp.findall("./skeleton"):
        if int(sk.get("frame")) == frame_id:
            sk_temporal = copy.deepcopy(sk)
            break
            
    # Step 2: Kinematic / Spatial Imputation
    sk_spatial = copy.deepcopy(sk_temporal)
    imputed_count = impute_one_skeleton(sk_spatial)
    print(f"Spatial imputed: {imputed_count} points")
    
    # Step 3: Calibration / Limb Length Clamping
    limb_lengths = calibrate_track_limb_lengths(track)
    sk_clamped = copy.deepcopy(sk_spatial)
    clamped_count = apply_limb_length_clamp(sk_clamped, limb_lengths, epsilon=5.0)
    print(f"Clamped: {clamped_count} segments")
    
    # Gather points maps
    map_orig = skeleton_points_map(sk_original)
    map_temp = skeleton_points_map(sk_temporal)
    map_spat = skeleton_points_map(sk_spatial)
    map_clam = skeleton_points_map(sk_clamped)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    plot_skeleton(axes[0], map_orig, f"1. Original\n(Missing: {17-len(map_orig)})")
    plot_skeleton(axes[1], map_temp, f"2. Temporal Interpolation\n(Missing: {17-len(map_temp)})")
    plot_skeleton(axes[2], map_spat, f"3. Spatial Imputation\n(Missing: {17-len(map_spat)})")
    plot_skeleton(axes[3], map_clam, f"4. Calibration & Clamp\n(Missing: {17-len(map_clam)})")
    
    fig.suptitle(f"Preprocessing Pipeline Effects - {zip_name} (Frame {frame_id})", fontsize=14)
    plt.tight_layout()
    output_path = "CapstoneProject_AIP491_FUHCM_Report_Template/preprocessing_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
