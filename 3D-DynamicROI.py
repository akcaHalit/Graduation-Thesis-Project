import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO

# --- Model ve Video Yolları ---
MODEL_PATH = "models/best.pt"
KASA_MODEL_PATH = "models/best-kasa.pt"
SVO_PATH = "videos/2/video.svo2"


# ==============================================================================
# Dinamik ROI Hesaplama Fonksiyonları (MP4 kodundan taşındı)
# ==============================================================================

def get_kasa_detections(frame: np.ndarray, kasa_model: YOLO) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Kasa tespiti yapar ve kutuları ile merkez koordinatlarını döndürür.
    Sadece "fish_box_custom" sınıfından olan kasaları döner.
    """
    results = kasa_model.predict(source=frame, conf=0.9, verbose=False)[0]
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


def compute_roi_direction_vector(frame_center: np.ndarray, kasa_centers: list[np.ndarray]) -> np.ndarray:
    """
    Görüntü merkezinden kasaların genel merkezine doğru birim yön vektörü hesaplar.
    Bu vektör, ROI'nin genel akış yönünü temsil eder.
    """
    if not kasa_centers:
        return None

    all_kasa_center = np.mean(kasa_centers, axis=0)
    roi_vec = frame_center - all_kasa_center
    norm = np.linalg.norm(roi_vec)
    if norm < 1e-6:  # Sıfıra yakın norma karşı koruma
        return None
    return roi_vec / norm


def find_base_edge_points(kasa_boxes: np.ndarray, kasa_centers: list[np.ndarray], roi_vec: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Kasaların, *sabit* ROI yön vektörüne en uygun "taban kenarını" oluşturan iki köşeyi bulur.
    Bu kenar, ROI'nin başlangıç konumunu belirlemek için kullanılır.
    """
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


def create_roi_polygon(pt1: np.ndarray, pt2: np.ndarray, roi_vec: np.ndarray,
                       offset: int = 20, depth: int = 150, extend: int = 70) -> np.ndarray:
    """
    Belirlenen taban kenarı ve *sabit* yön vektörüne göre ROI çokgenini oluşturur.
    Bu fonksiyon artık "dinamik" bir ROI oluşturmuyor, sadece sabit bir kalıp oluşturuyor.
    """
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


def point_in_polygon(cx: int, cy: int, polygon_pts: np.ndarray) -> bool:
    """
    Bir noktanın (cx, cy) bir çokgenin (polygon_pts) içinde olup olmadığını kontrol eder.
    """
    if polygon_pts is None or len(polygon_pts) == 0:
        return False
    point = (int(cx), int(cy))
    return cv2.pointPolygonTest(polygon_pts, point, False) >= 0


def point_in_box(cx, cy, box):
    """
    Bir noktanın (cx, cy) bir sınırlayıcı kutu (box) içinde olup olmadığını kontrol eder.
    """
    x1, y1, x2, y2 = map(int, box)
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ==============================================================================
# Ana İşleme Fonksiyonu
# ==============================================================================

def main():
    # --- Modelleri Yükle ---
    model = YOLO(MODEL_PATH)
    kasa_model = YOLO(KASA_MODEL_PATH)
    print(f"Balık modeli '{MODEL_PATH}' ve Kasa modeli '{KASA_MODEL_PATH}' yüklendi.")

    # --- ZED Kamerayı (SVO dosyasını) Başlatma ---
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(SVO_PATH)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED kamerası açılamadı veya SVO dosyası okunamadı: {repr(err)}")
        return

    camera_info = zed.get_camera_information()
    fps = camera_info.camera_configuration.fps
    print(f"SVO dosyası FPS: {fps}")

    start_second = 50
    if start_second > 0 and fps > 0:
        start_frame_index = int(start_second * fps)
        svo_length = zed.get_svo_number_of_frames()
        if start_frame_index >= svo_length:
            print(
                f"Uyarı: Başlangıç karesi ({start_frame_index}) SVO dosyasının sonundan ({svo_length}) büyük. Videonun başından başlanıyor.")
            start_frame_index = 0
        zed.set_svo_position(start_frame_index)
        print(f"SVO dosyası {start_second:.1f}. saniyeden ({start_frame_index}. kareden) başlatılıyor.")

    image_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    # --- Değişkenlerin Başlangıç Değerleri ---
    counted_ids = set()
    last_seen_positions = {}
    dropped_ids = set()
    kasa_counts = {}
    missing_counter = {}
    MISSING_THRESHOLD = 60
    frame_count = 0
    kasa_dict = {}

    # --- ROI ile İlgili Yeni Değişkenler ---
    fixed_roi_direction_vector = None  # Bir kez hesaplanır ve sabit kalır
    initial_kasa_avg_center = None  # İlk kasa grubunun ortalama merkezi
    initial_roi_polygon_template = None  # İlk kasa konumlarına göre oluşturulan ROI kalıbı

    KASA_RECALC_INTERVAL = 100  # Kaç frame'de bir kasaları yeniden tespit et ve ROI'yi kaydır

    # Parametreler (Sabitlendi)
    ROI_OFFSET = 10
    ROI_DEPTH = 100
    ROI_EXTEND = 70

    # --- Ana Döngü: Kareleri Oku ve İşle ---
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            current_frame_ocv = image_zed.get_data()
            if current_frame_ocv.shape[2] == 4:
                current_frame_ocv = cv2.cvtColor(current_frame_ocv, cv2.COLOR_BGRA2RGB)

            image_for_yolo = current_frame_ocv.copy()
            display_frame = cv2.cvtColor(current_frame_ocv, cv2.COLOR_RGB2BGR)
            frame_count += 1

            height, width, _ = display_frame.shape
            frame_center = np.array([width / 2, height / 2])

            # --- ROI Mantığı: Sabit Yön, Dinamik Konum ---
            current_roi_polygon = None

            # Sadece ilk frame'de (veya bulunana kadar) sabit yön vektörünü ve ROI kalıbını hesapla
            if initial_roi_polygon_template is None:
                initial_kasa_boxes, initial_kasa_centers = get_kasa_detections(image_for_yolo, kasa_model)

                if len(initial_kasa_centers) >= 2:
                    fixed_roi_direction_vector = compute_roi_direction_vector(frame_center, initial_kasa_centers)

                    if fixed_roi_direction_vector is not None:
                        initial_kasa_avg_center = np.mean(initial_kasa_centers, axis=0)

                        pt1_template, pt2_template = find_base_edge_points(
                            initial_kasa_boxes, initial_kasa_centers, fixed_roi_direction_vector
                        )

                        initial_roi_polygon_template = create_roi_polygon(
                            pt1_template, pt2_template, fixed_roi_direction_vector,
                            ROI_OFFSET, ROI_DEPTH, ROI_EXTEND
                        )

                        if initial_roi_polygon_template is not None:
                            print(
                                f"Frame {frame_count}: Sabit ROI yön vektörü ve ROI kalıbı oluşturuldu ve kilitlendi.")
                            sorted_kasa_info = sorted(zip(initial_kasa_boxes, initial_kasa_centers),
                                                      key=lambda x: x[0][0])
                            kasa_dict = {i: box for i, (box, _) in enumerate(sorted_kasa_info)}
                            for kasa_id in kasa_dict.keys():
                                if kasa_id not in kasa_counts:
                                    kasa_counts[kasa_id] = 0
                        else:
                            print(
                                f"Frame {frame_count}: ROI kalıbı oluşturulamadı (taban kenarı veya yön vektörü eksik). Tekrar denenecek.")
                    else:
                        print(
                            f"Frame {frame_count}: Yön vektörü oluşturulamadı (kasalar aynı noktada). Tekrar denenecek.")
                else:
                    print(
                        f"Frame {frame_count}: İlk frame'de yeterli kasa bulunamadı ({len(initial_kasa_centers)}). ROI kalıbı oluşturulamadı.")

            # Eğer sabit yön vektörü ve ROI kalıbı belirlenmişse, kasaları tespit et ve ROI'yi kaydır
            if initial_roi_polygon_template is not None:
                # Kasaları her KASA_RECALC_INTERVAL frame'de bir yeniden tespit et
                if frame_count == 1 or (frame_count % KASA_RECALC_INTERVAL == 0):  # % kullanımı daha doğru
                    current_kasa_boxes, current_kasa_centers = get_kasa_detections(image_for_yolo, kasa_model)

                    if len(current_kasa_boxes) > 0:
                        sorted_kasa_info = sorted(zip(current_kasa_boxes, current_kasa_centers), key=lambda x: x[0][0])
                        kasa_dict = {i: box for i, (box, _) in enumerate(sorted_kasa_info)}

                        for kasa_id in kasa_dict.keys():
                            if kasa_id not in kasa_counts:
                                kasa_counts[kasa_id] = 0
                        # last_kasa_detection_frame'e gerek kalmadı çünkü % ile kontrol ediyoruz
                        # print(f"Frame {frame_count}: Kasalar güncellendi.")
                    else:
                        kasa_dict = {}
                        # print(f"Frame {frame_count}: Kasa bulunamadı.")

                # Güncel kasa merkezlerinin ortalamasını al
                if len(kasa_dict) > 0:
                    current_kasa_centers_for_avg = [
                        np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                        for box in kasa_dict.values()
                    ]
                    current_kasa_avg_center = np.mean(current_kasa_centers_for_avg, axis=0)

                    # ROI kalıbını kaydır
                    offset_vector = current_kasa_avg_center - initial_kasa_avg_center
                    current_roi_polygon = initial_roi_polygon_template + offset_vector.astype(np.int32)
                    # print(f"Frame {frame_count}: ROI konumu güncellendi (kaydırıldı).")
                else:
                    current_roi_polygon = None
                    print(f"Frame {frame_count}: Güncel kasa bulunamadığı için ROI gösterilemiyor.")
            else:
                current_roi_polygon = None
                cv2.putText(display_frame, "Sabit ROI kalibi bekleniyor...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # --- ROI Mantığı Bitişi ---

            # Eğer bir güncel ROI varsa işleme devam et
            if current_roi_polygon is not None:
                # Balıkları takip et
                results = model.track(image_for_yolo, conf=0.35, persist=True, verbose=False)[0]

                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []
                confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []

                active_ids = set()

                for box, obj_id, conf in zip(boxes, ids, confs):
                    obj_id = int(obj_id)
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    active_ids.add(obj_id)

                    # Nokta GÜNCEL ROI içinde mi?
                    if point_in_polygon(cx, cy, current_roi_polygon):
                        if obj_id not in counted_ids:
                            counted_ids.add(obj_id)
                            # print(f"Balık {obj_id} ROI'ye girdi, sayıldı.")

                    if obj_id in counted_ids:
                        missing_counter[obj_id] = 0
                        last_seen_positions[obj_id] = (cx, cy)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = f"ID:{obj_id} Conf:{conf:.2f}"
                        cv2.putText(display_frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kayıp ID'leri güncelle ve Takipten çıkanları kasaya say
                for obj_id in list(last_seen_positions.keys()):
                    if obj_id not in active_ids:
                        missing_counter[obj_id] = missing_counter.get(obj_id, 0) + 1

                    if missing_counter.get(obj_id, 0) >= MISSING_THRESHOLD and obj_id not in dropped_ids:
                        cx, cy = last_seen_positions[obj_id]

                        assigned_to_any_box = False
                        for kasa_id, box in kasa_dict.items():
                            if point_in_box(cx, cy, box):
                                kasa_counts[kasa_id] = kasa_counts.get(kasa_id, 0) + 1
                                print(f"Balık {obj_id} -> Kasa {kasa_id}")
                                assigned_to_any_box = True
                                break

                        if assigned_to_any_box:
                            cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)
                            dropped_ids.add(obj_id)
                        elif obj_id not in dropped_ids:
                            pass  # Eğer hiçbir kasaya düşmediyse ama hala kayıpsa

                # Güncel ROI kutusunu çiz
                cv2.polylines(display_frame, [current_roi_polygon], True, (255, 0, 255), 2)

                # Toplam balık sayısını göster
                cv2.putText(display_frame, f"Total Fish Count: {len(counted_ids)}",
                            (width - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                cv2.putText(display_frame, "ROI gosterilemiyor...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Kasa kutularını çiz
            for _, box in kasa_dict.items():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                kasa_label = f"Kasa {kasa_id}"
                cv2.putText(display_frame, kasa_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Kasa sayaçlarını sol üst köşede sırayla yaz
            base_y = 40
            for kasa_id in sorted(kasa_counts.keys()):
                text = f"Kasa {kasa_id}: {kasa_counts[kasa_id]}"
                cv2.putText(display_frame, text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                base_y += 30

            cv2.imshow(f"ZED SVO Fish Counter - {SVO_PATH.split('/')[-1]}", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            print(f"SVO dosyasının sonuna ulaşıldı veya bir hata oluştu. Toplam {frame_count} kare işlendi.")
            break

    # --- Kapanış İşlemleri ---
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()