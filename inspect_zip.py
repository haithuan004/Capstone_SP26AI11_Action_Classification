import zipfile
import xml.etree.ElementTree as ET

zip_path = r"d:\Capstone2026\Action Predict\Labeled_data\task_2260379_annotations_2026_05_21_18_59_58_cvat for images 1.1_backup.zip"
with zipfile.ZipFile(zip_path, "r") as zf:
    xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
    with zf.open(xml_name) as f:
        tree = ET.parse(f)
        root = tree.getroot()

images = root.findall("./image")
actions_seen = []
track_id_counter = 0

# Let's see what attributes skeletons have
for img in images[:10]:
    sks = img.findall("./skeleton")
    for sk in sks:
        attrs = {a.get("name"): a.text for a in sk.findall("./attribute")}
        print(f"Image {img.get('id')}, Skeleton attributes: {attrs}")

# Count contiguous actions
current_action = None
action_blocks = []
count = 0
for img in images:
    sks = img.findall("./skeleton")
    if sks:
        attrs = {a.get("name"): a.text for a in sk.findall("./attribute")}
        action = attrs.get("action")
        if action != current_action:
            if current_action is not None:
                action_blocks.append((current_action, count))
            current_action = action
            count = 1
        else:
            count += 1
if current_action is not None:
    action_blocks.append((current_action, count))

print("Contiguous action blocks:")
for a, c in action_blocks:
    print(f"Action: {a}, Frames: {c}")
