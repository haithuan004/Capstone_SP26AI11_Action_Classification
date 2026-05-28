import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import os
import shutil

def process_xml(root):
    max_id = -1
    for t in root.findall('./track'):
        try:
            idx = int(t.get('id', '-1'))
            if idx > max_id:
                max_id = idx
        except:
            pass

    tracks = root.findall('./track')
    for t in tracks:
        skeletons = t.findall('./skeleton')
        num_frames = len(skeletons)
        
        if num_frames < 5:
            # Xoá track < 5 frames
            root.remove(t)
            
        elif num_frames > 60:
            # Xoá track gốc khỏi root
            root.remove(t)
            
            # Chia thành các chunk tối đa 50
            chunks = []
            curr_chunk = []
            for sk in skeletons:
                curr_chunk.append(sk)
                if len(curr_chunk) == 50:
                    chunks.append(curr_chunk)
                    curr_chunk = []
            
            # Nếu phần dư >= 5 frames thì giữ lại, < 5 thì bỏ
            if curr_chunk and len(curr_chunk) >= 5:
                chunks.append(curr_chunk)
                
            for chunk in chunks:
                max_id += 1
                new_track = ET.Element('track')
                
                # Sao chép các thuộc tính (vd: label, source...)
                for k, v in t.attrib.items():
                    if k == 'id':
                        new_track.set('id', str(max_id))
                    else:
                        new_track.set(k, v)
                
                if 'id' not in t.attrib:
                    new_track.set('id', str(max_id))
                    
                # Thêm các skeleton vào track mới
                for sk in chunk:
                    new_track.append(sk)
                    
                root.append(new_track)

def clean_all_zips(input_dir, output_dir, exclude_file):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zips = input_dir.glob('*.zip')
    for zp in zips:
        if zp.name == exclude_file:
            print(f"Skipping excluded file: {zp.name}")
            continue
            
        print(f"Processing: {zp.name}")
        out_zip = output_dir / zp.name
        
        with zipfile.ZipFile(zp, 'r') as zf_in:
            with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf_out:
                for item in zf_in.infolist():
                    if item.filename.endswith('.xml'):
                        with zf_in.open(item.filename) as xml_file:
                            tree = ET.parse(xml_file)
                            root = tree.getroot()
                            
                            # Xử lý logic cắt/lọc track
                            process_xml(root)
                            
                            # Ghi lại XML mới
                            xml_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                            zf_out.writestr(item, xml_bytes)
                    else:
                        # Copy nguyên các file khác (nếu có)
                        zf_out.writestr(item, zf_in.read(item.filename))
        
if __name__ == '__main__':
    in_dir = r'd:\Capstone2026\Action Predict\Labeled_data\Data_origin'
    out_dir = r'd:\Capstone2026\Action Predict\Labeled_data\Data_origin_cleaned'
    exclude = 'task_2260379_annotations_2026_05_21_18_59_58_cvat for images 1.1.zip'
    
    print("Starting process...")
    clean_all_zips(in_dir, out_dir, exclude)
    print(f"Done! Files saved at {out_dir}")
