import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
from itertools import combinations
import os

MODEL_PATH = "models/best.pt"
KASA_MODEL_PATH = "models/best-kasa.pt"

# Yardımcı fonksiyonlar (aynen kalıyor)
def get_kasa_detections(frame, kasa_model):
    results = kasa_model.predict(source=frame, conf=0.7, verbose=False)[0]
    boxes = []
    kasa_centers = []
    for box_r in results.boxes:
        detected_class_name = kasa_model.names[int(box_r.cls.cpu().numpy()[0])]
        if detected_class_name == "fish_box_custom":
            box = box_r.xyxy.cpu().numpy()[0]
            boxes.append(box)
            x1, y1, x2, y2 = box.astype(int)
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            kasa_centers.append(center)
    return np.array(boxes), kasa_centers

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
    for i in range(min(len(kasa_boxes), 3)):
        x1, y1, x2, y2 = kasa_boxes[i].astype(int)
        box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
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

def process(input_path, output_txt_path):
    output_lines = []
    model = YOLO(MODEL_PATH)
    kasa_model = YOLO(KASA_MODEL_PATH)
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(input_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        output_lines.append(f"ZED error: {repr(err)}")
        with open(output_txt_path + ".txt", "w") as f:
            f.write("\n".join(output_lines))
        return

    image_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    counted_ids = set()
    dropped_ids = set()
    last_seen_positions = {}
    missing_counter = {}
    kasa_counts = {}
    frame_count = 0
    MISSING_THRESHOLD = 10
    ROI_OFFSET, ROI_DEPTH, ROI_EXTEND = 20, 150, 70

    fixed_roi_direction_vector = None
    initial_kasa_avg_center = None
    initial_roi_polygon_template = None
    kasa_dict = {}

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            break

        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame = image_zed.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        height, width, _ = frame.shape
        frame_center = np.array([width / 2, height / 2])

        frame_count += 1

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

        if initial_roi_polygon_template is None:
            continue

        current_kasa_centers = [
            np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) for box in kasa_dict.values()
        ]
        offset_vector = np.mean(current_kasa_centers, axis=0) - initial_kasa_avg_center
        current_roi_polygon = initial_roi_polygon_template + offset_vector.astype(np.int32)

        results = model.track(frame, conf=0.3, persist=True)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []
        active_ids = set()

        for box, obj_id in zip(boxes, ids):
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

        for obj_id in list(last_seen_positions.keys()):
            if obj_id not in active_ids:
                missing_counter[obj_id] = missing_counter.get(obj_id, 0) + 1
            if missing_counter[obj_id] >= MISSING_THRESHOLD and obj_id not in dropped_ids:
                cx, cy = last_seen_positions[obj_id]
                for kasa_id, box in kasa_dict.items():
                    if point_in_box(cx, cy, box):
                        kasa_counts[kasa_id] += 1
                        output_lines.append(f"Frame {frame_count} - Fish ID {obj_id} -> Box {kasa_id}")
                        break
                dropped_ids.add(obj_id)

    zed.close()

    output_lines.append("\nFinal Counts:")
    for kasa_id in sorted(kasa_counts.keys()):
        output_lines.append(f"Kasa {kasa_id}: {kasa_counts[kasa_id]}")

    with open(output_txt_path + ".txt", "w") as f:
        f.write("\n".join(output_lines))
