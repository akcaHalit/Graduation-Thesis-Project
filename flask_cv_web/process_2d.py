import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
KASA_MODEL_PATH = "models/best-kasa.pt"

# — Yardımcı fonksiyonlar (değişmedi) —
# get_kasa_detections
# compute_roi_direction_vector
# find_base_edge_points
# create_roi_polygon
# point_in_polygon
# point_in_box

# Yukarıdaki fonksiyonları sen zaten gönderdin, aynen bırakıyoruz

def get_kasa_detections(frame, kasa_model):
    results = kasa_model.predict(source=frame, conf=0.9, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    kasa_centers = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        kasa_centers.append(center)
    return boxes, kasa_centers

def compute_roi_direction_vector(frame_center, kasa_centers):
    if not kasa_centers:
        return None
    all_kasa_center = np.mean(kasa_centers, axis=0)
    roi_vec = frame_center - all_kasa_center
    norm = np.linalg.norm(roi_vec)
    if norm < 1e-6:
        return None
    return roi_vec / norm

def find_base_edge_points(kasa_boxes, kasa_centers, roi_vec):
    if len(kasa_boxes) < 2:
        return None, None
    scores = []
    for i in range(min(2, len(kasa_boxes))):
        x1, y1, x2, y2 = kasa_boxes[i].astype(int)
        box_pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        kasa_center = kasa_centers[i]
        for pt in box_pts:
            vec = pt - kasa_center
            vec_norm = vec / (np.linalg.norm(vec) + 1e-6)
            sim = np.dot(vec_norm, roi_vec)
            scores.append((sim, pt))
    top_pts = sorted(scores, key=lambda x: x[0], reverse=True)[:4]
    if len(top_pts) < 2:
        return None, None
    max_dist = -1
    pt1, pt2 = None, None
    for i in range(len(top_pts)):
        for j in range(i + 1, len(top_pts)):
            dist = np.linalg.norm(top_pts[i][1] - top_pts[j][1])
            if dist > max_dist:
                max_dist = dist
                pt1, pt2 = top_pts[i][1], top_pts[j][1]
    if pt1 is None or pt2 is None:
        return None, None
    if pt1[0] > pt2[0]:
        pt1, pt2 = pt2, pt1
    return pt1, pt2

def create_roi_polygon(pt1, pt2, roi_vec, offset=20, depth=150, extend=70):
    if pt1 is None or pt2 is None or roi_vec is None:
        return None
    edge_vec = pt2 - pt1
    edge_vec_norm = edge_vec / (np.linalg.norm(edge_vec) + 1e-6)
    pt1_ext = pt1 - edge_vec_norm * extend
    pt2_ext = pt2 + edge_vec_norm * extend
    normal_vec = np.array([-edge_vec_norm[1], edge_vec_norm[0]])
    if np.dot(normal_vec, roi_vec) < 0:
        normal_vec = -normal_vec
    pt1_shifted = pt1_ext + normal_vec * offset
    pt2_shifted = pt2_ext + normal_vec * offset
    pt3 = pt2_shifted + normal_vec * depth
    pt4 = pt1_shifted + normal_vec * depth
    return np.array([pt1_shifted, pt2_shifted, pt3, pt4], dtype=np.int32)

def point_in_polygon(cx, cy, polygon_pts):
    if polygon_pts is None or len(polygon_pts) == 0:
        return False
    point = (int(cx), int(cy))
    return cv2.pointPolygonTest(polygon_pts, point, False) >= 0

def point_in_box(cx, cy, box):
    x1, y1, x2, y2 = map(int, box)
    return x1 <= cx <= x2 and y1 <= cy <= y2

# ————————————————————————
# Ana Fonksiyon: Flask çağırır
# ————————————————————————

def process(input_path, output_path):
    model = YOLO(MODEL_PATH)
    kasa_model = YOLO(KASA_MODEL_PATH)
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Video açılamadı:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(fps * 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fixed_roi_direction_vector = None
    initial_kasa_avg_center = None
    initial_roi_polygon_template = None
    counted_ids = set()
    last_seen_positions = {}
    dropped_ids = set()
    kasa_counts = {}
    missing_counter = {}
    MISSING_THRESHOLD = 5
    frame_count = 0
    kasa_dict = {}
    ROI_OFFSET = 20
    ROI_DEPTH = 150
    ROI_EXTEND = 70
    KASA_RECALC_INTERVAL = 60

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_center = np.array([width / 2, height / 2])

        if initial_roi_polygon_template is None:
            initial_kasa_boxes, initial_kasa_centers = get_kasa_detections(frame, kasa_model)
            if len(initial_kasa_centers) >= 2:
                fixed_roi_direction_vector = compute_roi_direction_vector(frame_center, initial_kasa_centers)
                if fixed_roi_direction_vector is not None:
                    initial_kasa_avg_center = np.mean(initial_kasa_centers, axis=0)
                    pt1, pt2 = find_base_edge_points(initial_kasa_boxes, initial_kasa_centers, fixed_roi_direction_vector)
                    initial_roi_polygon_template = create_roi_polygon(pt1, pt2, fixed_roi_direction_vector,
                                                                      ROI_OFFSET, ROI_DEPTH, ROI_EXTEND)
                    sorted_kasa_info = sorted(zip(initial_kasa_boxes, initial_kasa_centers), key=lambda x: x[0][0])
                    kasa_dict = {i: box for i, (box, _) in enumerate(sorted_kasa_info)}
                    for kasa_id in kasa_dict.keys():
                        kasa_counts[kasa_id] = 0

        current_roi_polygon = None
        if initial_roi_polygon_template is not None:
            if frame_count == 1 or (frame_count % KASA_RECALC_INTERVAL == 0):
                current_kasa_boxes, current_kasa_centers = get_kasa_detections(frame, kasa_model)
                if len(current_kasa_boxes) > 0:
                    sorted_kasa_info = sorted(zip(current_kasa_boxes, current_kasa_centers), key=lambda x: x[0][0])
                    kasa_dict = {i: box for i, (box, _) in enumerate(sorted_kasa_info)}
                    for kasa_id in kasa_dict.keys():
                        if kasa_id not in kasa_counts:
                            kasa_counts[kasa_id] = 0
            if len(kasa_dict) > 0:
                current_kasa_centers = [
                    np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    for box in kasa_dict.values()
                ]
                offset_vector = np.mean(current_kasa_centers, axis=0) - initial_kasa_avg_center
                current_roi_polygon = initial_roi_polygon_template + offset_vector.astype(np.int32)

        if current_roi_polygon is not None:
            results = model.track(frame, persist=True)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []
            active_ids = set()
            confs = results.boxes.conf.cpu().numpy()

            """for box, obj_id in zip(boxes, ids):
                obj_id = int(obj_id)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                active_ids.add(obj_id)
                if point_in_polygon(cx, cy, current_roi_polygon):
                    if obj_id not in counted_ids:
                        counted_ids.add(obj_id)
                if obj_id in counted_ids:
                    missing_counter[obj_id] = 0
                    last_seen_positions[obj_id] = (cx, cy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"""
            for box, obj_id, conf in zip(boxes, ids, confs):
                obj_id = int(obj_id)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                active_ids.add(obj_id)

                if point_in_polygon(cx, cy, current_roi_polygon):
                    if obj_id not in counted_ids:
                        counted_ids.add(obj_id)

                if obj_id in counted_ids:
                    missing_counter[obj_id] = 0
                    last_seen_positions[obj_id] = (cx, cy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 🆕 Conf değeriyle birlikte yazı
                    label = f"ID:{obj_id} | {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)


            for obj_id in list(last_seen_positions.keys()):
                if obj_id not in active_ids:
                    missing_counter[obj_id] = missing_counter.get(obj_id, 0) + 1

            for obj_id in list(last_seen_positions.keys()):
                if (obj_id in counted_ids and
                        missing_counter.get(obj_id, 0) >= MISSING_THRESHOLD and
                        obj_id not in dropped_ids):
                    cx, cy = last_seen_positions[obj_id]
                    for kasa_id, box in kasa_dict.items():
                        if point_in_box(cx, cy, box):
                            kasa_counts[kasa_id] += 1
                            dropped_ids.add(obj_id)
                            break
                    dropped_ids.add(obj_id)

            cv2.polylines(frame, [current_roi_polygon], True, (255, 0, 255), 2)

        """for _, box in kasa_dict.items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 💬 Kasaya ID yazısı ekle
            label = f"Kasa {kasa_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)"""
        for kasa_id, box in kasa_dict.items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 💬 Her kutuya kendi ID’si yazılır
            label = f"Kasa {kasa_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

        # ✅ Sayım bilgilerini frame’e ekle (SOL ÜST)
        base_y = 40
        for kasa_id in sorted(kasa_counts.keys()):
            text = f"Kasa {kasa_id}: {kasa_counts[kasa_id]}"
            cv2.putText(frame, text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
            base_y += 30

        # ✅ Frame’i diske yaz
        out.write(frame)

    cap.release()
    out.release()
