import os
import json

# Girdi klasörü (LabelMe JSON dosyaları burada)
input_dir = "../data/labels"

# Çıktı klasörü (YOLO txt dosyaları burada olacak)
output_dir = "../data/k-fold/labels_all"
os.makedirs(output_dir, exist_ok=True)

# Dosyaları işle
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(input_dir, filename)

        with open(json_path, "r") as f:
            data = json.load(f)

        image_w = data.get("imageWidth")
        image_h = data.get("imageHeight")
        shapes = data.get("shapes", [])

        yolo_lines = []

        for shape in shapes:
            if shape.get("shape_type") != "rectangle":
                continue

            (x1, y1), (x2, y2) = shape["points"]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            x_center = ((x_min + x_max) / 2) / image_w
            y_center = ((y_min + y_max) / 2) / image_h
            width = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h

            class_id = 0  # Tek sınıf: S_aurata

            yolo_lines.append(f"{class_id} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}")

        # .txt olarak kaydet
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("✅ Tüm JSON dosyaları YOLO formatında 'data/labels' klasörüne yazıldı.")
