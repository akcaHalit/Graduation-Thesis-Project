import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
KASA_MODEL_PATH = "models/best-kasa.pt"
VIDEO_PATH = "videos/at.mp4"

def select_roi(window_name: str, frame: np.ndarray) -> tuple[int,int,int,int]:
    roi = cv2.selectROI(window_name, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        raise RuntimeError("Geçerli bir ROI seçilmedi.")
    return x, y, w, h

def point_in_box(cx, cy, box):
    x1, y1, x2, y2 = map(int, box)
    return x1 <= cx <= x2 and y1 <= cy <= y2

def main():
    model = YOLO(MODEL_PATH)
    kasa_model = YOLO(KASA_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Video açılamadı:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    roi = None
    counted_ids = set()
    last_seen_positions = {}
    dropped_ids = set()
    kasa_counts = {}
    missing_counter = {}  # 🔁 ID: kaç frame'dir kayıp?
    MISSING_THRESHOLD = 5  # 🔢 Kaç frame boyunca yoksa artık düştü kabul edelim?
    frame_count = 0
    kasa_dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            try:
                roi = select_roi("ROI Seç", frame)
                counted_ids.clear()
                dropped_ids.clear()
                last_seen_positions.clear()
                print("Yeni ROI:", roi)
            except RuntimeError as e:
                print(e)
                roi = None

        if roi is not None:
            x, y, w, h = roi

            # Kasaları tespit ve sırala
            # İlk karede VEYA her 100 frame'de bir kasa tespiti
            if frame_count == 0 or frame_count % 100 == 0:
                kasa_results = kasa_model.track(frame, conf=0.9, persist=True)[0]
                kasa_boxes = kasa_results.boxes.xyxy.cpu().numpy()

                if len(kasa_boxes) > 0:
                    kasa_boxes_sorted = sorted(kasa_boxes, key=lambda b: b[0])
                    kasa_dict = {i: box for i, box in enumerate(kasa_boxes_sorted)}

            for kasa_id in kasa_dict:
                if kasa_id not in kasa_counts:
                    kasa_counts[kasa_id] = 0

            # Balıkları takip et
            results = model.track(frame, persist=True)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else  []
            confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []

            active_ids = set()

            #for box, obj_id in zip(boxes, ids):
            for box, obj_id, conf in zip(boxes, ids, confs):
                obj_id = int(obj_id)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                active_ids.add(obj_id)

                if x <= cx <= x + w and y <= cy <= y + h:
                    if obj_id not in counted_ids:
                        counted_ids.add(obj_id)

                if obj_id in counted_ids:
                    missing_counter[obj_id] = 0  # 👈 Göründü, sıfırla
                    last_seen_positions[obj_id] = (cx, cy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    """cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"""
                    cv2.putText(frame, f"ID:{obj_id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 🔴 KAYIP ID'LERİ GÜNCELLE
            for obj_id in list(last_seen_positions.keys()):
                if obj_id not in active_ids:
                    missing_counter[obj_id] = missing_counter.get(obj_id, 0) + 1

            # Takipten çıkanlar için kasaya sayım
            for obj_id in list(last_seen_positions.keys()):
                if missing_counter.get(obj_id, 0) >= MISSING_THRESHOLD and obj_id not in dropped_ids:
                    cx, cy = last_seen_positions[obj_id]

                    for kasa_id, box in kasa_dict.items():
                        if point_in_box(cx, cy, box):
                            kasa_counts[kasa_id] += 1
                            print(f"Balık {obj_id} -> Kasa {kasa_id}")
                            break

                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    dropped_ids.add(obj_id)

            # ROI kutusu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, f"Toplam Sayilan Balik: {len(counted_ids)}",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Kasa kutularını çiz ama yazı yazma
            for _, box in kasa_dict.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 🟨 Kasa sayaçlarını sol üst köşede sırayla yaz
            base_y = 40
            for kasa_id in sorted(kasa_counts.keys()):
                text = f"Kasa {kasa_id}: {kasa_counts[kasa_id]}"
                cv2.putText(frame, text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                base_y += 30  # alt satıra geç

        cv2.imshow("Fish Counter", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
