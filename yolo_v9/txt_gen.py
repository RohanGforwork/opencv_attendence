import os


base_dir = "persons"

output_dir = "yolo_dataset"
os.makedirs(output_dir, exist_ok=True)

class_names = sorted(os.listdir(base_dir))
class_to_id = {cls: i for i, cls in enumerate(class_names)}
print(f"Class to ID Mapping: {class_to_id}")

for class_name in class_names:
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)

        label_path = os.path.splitext(image_path.replace(base_dir, output_dir))[0] + ".txt"
        label_dir = os.path.dirname(label_path)
        os.makedirs(label_dir, exist_ok=True)

        with open(label_path, "w") as label_file:
            class_id = class_to_id[class_name]

            label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("All .txt files generated with dummy labels!")
