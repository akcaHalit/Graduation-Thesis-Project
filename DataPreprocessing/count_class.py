import os
import json

input_dir = "../data/labels"
all_labels = set()

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r") as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label:
                    all_labels.add(label)

# Sonuçları göster
print(f"Toplam sınıf sayısı: {len(all_labels)}")
print("Sınıf isimleri:")
for i, label in enumerate(sorted(all_labels)):
    print(f"{i}: {label}")



#Toplam sınıf sayısı: 1
#Sınıf isimleri:
#0: S_aurata


