import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil

CLASSES = [
    'bike', 'bus', 'car', 'motor', 'person', 
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]

def create_voc_xml(img_path, txt_path, xml_path, folder_name):
    img = cv2.imread(img_path)
    if img is None:
        return False
    height, width, depth = img.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                class_id = int(parts[0])
                class_name = CLASSES[class_id]
                
                x_center, y_center, w, h = map(float, parts[1:5])
                
                xmin = int((x_center - (w / 2)) * width)
                ymin = int((y_center - (h / 2)) * height)
                xmax = int((x_center + (w / 2)) * width)
                ymax = int((y_center + (h / 2)) * height)

                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(width - 1, xmax)
                ymax = min(height - 1, ymax)

                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = class_name
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(xmin)
                ET.SubElement(bndbox, 'ymin').text = str(ymin)
                ET.SubElement(bndbox, 'xmax').text = str(xmax)
                ET.SubElement(bndbox, 'ymax').text = str(ymax)

    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(xml_path, 'w') as f:
        f.write(xmlstr)
    
    return True

def main():
    #Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_dir = os.path.join(root_dir, "data", "master_dataset")
    voc_dir = os.path.join(root_dir, "data", "master_voc_dataset")

    print("Starting PASCAL VOC Conversion")

    for split in ['train', 'valid']:
        img_src_dir = os.path.join(yolo_dir, split, "images")
        lbl_src_dir = os.path.join(yolo_dir, split, "labels")
        
        img_dst_dir = os.path.join(voc_dir, split, "JPEGImages")
        xml_dst_dir = os.path.join(voc_dir, split, "Annotations")
        
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(xml_dst_dir, exist_ok=True)

        if not os.path.exists(img_src_dir): continue

        images = os.listdir(img_src_dir)
        print(f"Converting {len(images)} images in {split} split")

        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_src_dir, img_name)
            txt_path = os.path.join(lbl_src_dir, f"{base_name}.txt")
            xml_path = os.path.join(xml_dst_dir, f"{base_name}.xml")
            #Copy image
            shutil.copy(img_path, os.path.join(img_dst_dir, img_name))
            
            #Generate XML
            create_voc_xml(img_path, txt_path, xml_path, split)

    print(f"\nConversion Complete")

if __name__ == "__main__":
    main()