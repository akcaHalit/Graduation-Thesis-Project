# ETİKETLENMİŞ VERİYİ (Serkan Hocadan aldığım) =>  train-val-test olarak ayırdım. 0.7 / 0.2 / 0.1  => EN İYİ ÇALIŞAN MODEL ve DATASET oldu.

import os
import shutil
import random

# Klasör Oranları
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Klasör yolları
base_dir = "etiketli_dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

output_base = "dataset_split"
subsets = ["train", "val", "test"]

# Resim dosyalarını topla
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# Bölme noktaları
total = len(image_files)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

# Dosyaları kopyala
for subset in subsets:
    for subfolder in ['images', 'labels']:
        os.makedirs(os.path.join(output_base, subset, subfolder), exist_ok=True)

    for img_file in splits[subset]:
        label_file = os.path.splitext(img_file)[0] + ".txt"

        shutil.copy(os.path.join(images_dir, img_file), os.path.join(output_base, subset, "images", img_file))
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_base, subset, "labels", label_file))
