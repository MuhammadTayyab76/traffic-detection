import os
import shutil

target_classes = [
    'bike', 'bus', 'car', 'motor', 'person', 
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]

bdd_map = {
    0: 0,   #bike -> bike
    1: 1,   #bus -> bus
    2: 2,   # car -> car
    5: 3,   #motor -> motor
    6: 4,   #person -> person
    7: 5,   #rider -> rider
    8: 6,   #traffic light -> traffic light
    9: 7,   #traffic sign -> traffic sign
    10: 8,  #train -> train
    11: 9   #truck -> truck
}

detrac_map = {
    0: 2,   
    1: 1,   
    2: 9    
}

def process_dataset(source_dir, dest_dir, class_map, split_name):
    os.makedirs(f"{dest_dir}/{split_name}/images", exist_ok=True)
    os.makedirs(f"{dest_dir}/{split_name}/labels", exist_ok=True)

    images_dir = f"{source_dir}/{split_name}/images"
    if not os.path.exists(images_dir):
        print(f"Skipping {images_dir} - Not found")
        return

    print(f"Processing {source_dir} - {split_name}...")
    
    for img_name in os.listdir(images_dir):
        base_name = os.path.splitext(img_name)[0]
        txt_name = f"{base_name}.txt"
        
        img_src = f"{images_dir}/{img_name}"
        txt_src = f"{source_dir}/{split_name}/labels/{txt_name}"

        if not os.path.exists(txt_src): 
            continue

        valid_lines = []
        with open(txt_src, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                old_id = int(parts[0])
                if old_id in class_map:
                    new_id = class_map[old_id]
                    parts[0] = str(new_id)
                    valid_lines.append(" ".join(parts))

        if valid_lines:
            shutil.copy(img_src, f"{dest_dir}/{split_name}/images/{img_name}")
            with open(f"{dest_dir}/{split_name}/labels/{txt_name}", 'w') as f:
                f.write("\n".join(valid_lines) + "\n")

def main():
    #Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    master_dir = os.path.join(root_dir, "data", "master_dataset")
    bdd_dir = os.path.join(root_dir, "data", "bdd100k")
    detrac_dir = os.path.join(root_dir, "data", "ua_detrac")

    print("Starting Dataset Merge...")

    for split in ['train', 'valid']:
        process_dataset(bdd_dir, master_dir, bdd_map, split)
        process_dataset(detrac_dir, master_dir, detrac_map, split)

    yaml_content = f"train: ../train/images\nval: ../valid/images\n\nnc: {len(target_classes)}\nnames: {target_classes}"
    with open(f"{master_dir}/data.yaml", 'w') as f:
        f.write(yaml_content)
        
    print(f"\nMerge Complete! Your master dataset is ready at: {master_dir}")

if __name__ == "__main__":
    main()