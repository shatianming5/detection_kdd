import os
import json
import xml.etree.ElementTree as ET
import glob
import shutil
import random
from tqdm import tqdm

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Ignore the filename in XML, derive from file path
    # Assuming matching image is .jpg
    file_id = os.path.splitext(os.path.basename(xml_file))[0]
    filename = file_id + ".jpg"
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Convert to COCO format [x, y, width, height]
        w = xmax - xmin
        h = ymax - ymin
        boxes.append({
            'category': name,
            'bbox': [xmin, ymin, w, h],
            'area': w * h
        })
        
    return {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': int(file_id) if file_id.isdigit() else hash(file_id) % ((sys.maxsize + 1) * 2), # Use int if possible, else hash
        'annotations': boxes
    }

def convert_to_coco(source_dir, target_dir, split_ratio=0.9):
    print(f"Converting {source_dir} to COCO format in {target_dir}...")
    
    annotations_dir = os.path.join(source_dir, "Annotations")
    images_dir = os.path.join(source_dir, "JPEGImages")
    
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    random.shuffle(xml_files)
    
    train_size = int(len(xml_files) * split_ratio)
    splits = {
        'train2017': xml_files[:train_size],
        'val2017': xml_files[train_size:]
    }
    
    # Prepare categories
    categories = set()
    
    # Pass 1: Collect categories (optional, but good for ID assignment)
    # We'll just build it dynamically
    category_map = {}
    cat_id_counter = 1
    
    if not os.path.exists(os.path.join(target_dir, "annotations")):
        os.makedirs(os.path.join(target_dir, "annotations"))

    for split_name, files in splits.items():
        print(f"Processing {split_name} ({len(files)} images)...")
        
        # Create image directory
        img_target_dir = os.path.join(target_dir, split_name)
        if not os.path.exists(img_target_dir):
            os.makedirs(img_target_dir)
            
        dataset = {
            "info": {
                "description": "Repro 10k Dataset",
                "url": "",
                "version": "1.0",
                "year": 2025,
                "contributor": "",
                "date_created": "2025/12/09"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        ann_id_counter = 1
        
        for xml_file in tqdm(files):
            try:
                img_info = parse_xml(xml_file)
                
                # Check if image exists
                src_img_path = os.path.join(images_dir, img_info['file_name'])
                if not os.path.exists(src_img_path):
                    print(f"Warning: Image {src_img_path} not found. Skipping.")
                    continue
                
                # Symlink image
                dst_img_path = os.path.join(img_target_dir, img_info['file_name'])
                if not os.path.exists(dst_img_path):
                    os.symlink(os.path.abspath(src_img_path), dst_img_path)
                
                dataset['images'].append({
                    "file_name": img_info['file_name'],
                    "height": img_info['height'],
                    "width": img_info['width'],
                    "id": img_info['id']
                })
                
                for ann in img_info['annotations']:
                    cat_name = ann['category']
                    if cat_name not in category_map:
                        category_map[cat_name] = cat_id_counter
                        cat_id_counter += 1
                    
                    dataset['annotations'].append({
                        "id": ann_id_counter,
                        "image_id": img_info['id'],
                        "category_id": category_map[cat_name],
                        "bbox": ann['bbox'],
                        "area": ann['area'],
                        "iscrowd": 0,
                        "segmentation": [] # Bbox only for now
                    })
                    ann_id_counter += 1
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

        # Add categories
        for name, id in category_map.items():
            dataset['categories'].append({"id": id, "name": name})
            
        # Save JSON
        json_path = os.path.join(target_dir, "annotations", f"instances_{split_name}.json")
        with open(json_path, 'w') as f:
            json.dump(dataset, f)
            
    print("Conversion complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python prepare_coco_data.py <voc_root> <output_dir>")
    else:
        convert_to_coco(sys.argv[1], sys.argv[2])
