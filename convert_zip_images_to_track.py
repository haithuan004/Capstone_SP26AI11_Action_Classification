import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import shutil

def convert_zip(zip_path: Path):
    print(f"Processing zip: {zip_path.name}")
    # 1. Read XML from zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
        if not xml_name:
            print(f"Error: No XML found in {zip_path.name}")
            return
        with zf.open(xml_name) as f:
            tree = ET.parse(f)
            root = tree.getroot()

    # 2. Check if tracks already exist
    if root.findall("./track"):
        print("Tracks already exist, skipping conversion.")
        return

    images = root.findall("./image")
    if not images:
        print("No images found, skipping conversion.")
        return

    print(f"Converting {len(images)} image elements to tracks...")
    
    # Create track element
    track = ET.Element("track", id="0", label="person", source="manual")

    for img in images:
        img_id = img.get("id")
        skeletons = img.findall("./skeleton")
        for sk in skeletons:
            # Create a track skeleton node
            track_sk = ET.Element("skeleton", frame=img_id, keyframe="1", z_order=sk.get("z_order", "0"))
            
            # Copy attributes (action, is_crowd)
            for attr in sk.findall("./attribute"):
                track_sk.append(attr)

            # Copy points
            for pt in sk.findall("./points"):
                pt_label = pt.get("label")
                pt_outside = pt.get("outside", "0")
                pt_occluded = pt.get("occluded", "0")
                pt_coords = pt.get("points")
                
                track_pt = ET.Element("points", label=pt_label, keyframe="1", outside=pt_outside, occluded=pt_occluded, points=pt_coords)
                track_sk.append(track_pt)

            track.append(track_sk)

    root.append(track)

    # 3. Create a temporary zip and replace
    temp_dir = Path(tempfile.mkdtemp())
    temp_zip_path = temp_dir / zip_path.name

    # Write converted XML bytes to zip
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    
    # Keep backup of original zip
    backup_path = zip_path.with_name(f"{zip_path.stem}_backup{zip_path.suffix}")
    if not backup_path.exists():
        shutil.copy2(zip_path, backup_path)
        print(f"Backup of original zip saved to {backup_path.name}")

    with zipfile.ZipFile(zip_path, "r") as src_zip, zipfile.ZipFile(
        temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst_zip:
        for info in src_zip.infolist():
            if info.filename == xml_name:
                dst_zip.writestr(info, xml_bytes)
            else:
                dst_zip.writestr(info, src_zip.read(info.filename))

    # Overwrite the original zip
    shutil.move(str(temp_zip_path), str(zip_path))
    shutil.rmtree(temp_dir)
    print(f"Successfully converted and updated {zip_path.name}\n")

if __name__ == "__main__":
    zip_p = Path("Data_origin/task_2260379_annotations_2026_05_21_18_59_58_cvat for images 1.1.zip")
    convert_zip(zip_p)
