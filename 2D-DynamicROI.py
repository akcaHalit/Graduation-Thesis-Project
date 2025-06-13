import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
KASA_MODEL_PATH = "models/best-kasa.pt"
VIDEO_PATH = "videos/video-coklu.mp4"


# ==============================================================================
# Yardımcı Fonksiyonlar
# ==============================================================================

def get_kasa_detections(frame: np.ndarray, kasa_model: YOLO) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Kasa tespiti yapar ve kutuları ile merkez koordinatlarını döndürür.
    """
    results = kasa_model.predict(source=frame, conf=0.95, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    kasa_centers = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        kasa_centers.append(center)
        # Hata ayıklama için kasaları çiz. Nihai kodda kaldırılabilir.
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(frame, f"{confs[i]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return boxes, kasa_centers


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
    model = YOLO(MODEL_PATH)
    kasa_model = YOLO(KASA_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Video açılamadı:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * 30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ROI sabitlenmesi için ilk kasa bilgileri ve yön vektörü
    fixed_roi_direction_vector = None  # Bir kez hesaplanır ve sabit kalır
    initial_kasa_avg_center = None  # İlk kasa grubunun ortalama merkezi
    initial_roi_polygon_template = None  # İlk kasa konumlarına göre oluşturulan ROI kalıbı

    last_kasa_detection_frame = -1  # En son kasa tespiti yapıldığı frame

    counted_ids = set()
    last_seen_positions = {}
    dropped_ids = set()
    kasa_counts = {}
    missing_counter = {}
    MISSING_THRESHOLD = 5
    frame_count = 0
    kasa_dict = {}  # Kasa ID'lerine göre kutuları tutar (sıralı)

    # Parametreler
    ROI_OFFSET = 10
    ROI_DEPTH = 175
    ROI_EXTEND = 80
    KASA_RECALC_INTERVAL = 75  # Kaç frame'de bir kasaları yeniden tespit et

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        height, width, _ = frame.shape
        frame_center = np.array([width / 2, height / 2])

        # Sadece ilk frame'de (veya bulunana kadar) sabit yön vektörünü ve ROI kalıbını hesapla
        if initial_roi_polygon_template is None:
            initial_kasa_boxes, initial_kasa_centers = get_kasa_detections(frame, kasa_model)

            if len(initial_kasa_centers) >= 2:  # Yön vektörü ve taban kenarı için en az 2 kasa idealdir
                fixed_roi_direction_vector = compute_roi_direction_vector(frame_center, initial_kasa_centers)

                if fixed_roi_direction_vector is not None:
                    # İlk kasa merkezlerinin ortalamasını al, bu ROI'nin "referans" merkezidir.
                    initial_kasa_avg_center = np.mean(initial_kasa_centers, axis=0)

                    # Bu sabit yön vektörü ve ilk kasa konumlarına göre bir "şablon" ROI oluştur
                    # Bu şablon, ROI'nin şeklini ve oryantasyonunu belirler.
                    pt1_template, pt2_template = find_base_edge_points(
                        initial_kasa_boxes, initial_kasa_centers, fixed_roi_direction_vector
                    )

                    initial_roi_polygon_template = create_roi_polygon(
                        pt1_template, pt2_template, fixed_roi_direction_vector,
                        ROI_OFFSET, ROI_DEPTH, ROI_EXTEND
                    )

                    if initial_roi_polygon_template is not None:
                        print(f"Frame {frame_count}: Sabit ROI yön vektörü ve ROI kalıbı oluşturuldu ve kilitlendi.")
                        # İlk tespit edilen kasaları da kasa_dict'e kaydet
                        sorted_kasa_info = sorted(zip(initial_kasa_boxes, initial_kasa_centers), key=lambda x: x[0][0])
                        kasa_dict = {i: box for i, (box, _) in enumerate(sorted_kasa_info)}
                        # Kasa sayaçlarını başlat
                        for kasa_id in kasa_dict.keys():
                            if kasa_id not in kasa_counts:
                                kasa_counts[kasa_id] = 0
                    else:
                        print(
                            f"Frame {frame_count}: ROI kalıbı oluşturulamadı (taban kenarı veya yön vektörü eksik). Tekrar denenecek.")
                else:
                    print(f"Frame {frame_count}: Yön vektörü oluşturulamadı (kasalar aynı noktada). Tekrar denenecek.")
            else:
                print(
                    f"Frame {frame_count}: İlk frame'de yeterli kasa bulunamadı ({len(initial_kasa_centers)}). ROI kalıbı oluşturulamadı.")

        # Eğer sabit yön vektörü ve ROI kalıbı belirlenmişse, kasaları tespit et ve ROI'yi kaydır
        current_roi_polygon = None  # Her frame için güncel ROI
        if initial_roi_polygon_template is not None:
            # Kasaları her 100 frame'de bir yeniden tespit et
            if frame_count == 1 or (frame_count - last_kasa_detection_frame >= KASA_RECALC_INTERVAL):
                current_kasa_boxes, current_kasa_centers = get_kasa_detections(frame, kasa_model)

                if len(current_kasa_boxes) > 0:
                    sorted_kasa_info = sorted(zip(current_kasa_boxes, current_kasa_centers), key=lambda x: x[0][0])
                    kasa_dict = {i: box for i, (box, _) in enumerate(sorted_kasa_info)}

                    for kasa_id in kasa_dict.keys():
                        if kasa_id not in kasa_counts:
                            kasa_counts[kasa_id] = 0
                    last_kasa_detection_frame = frame_count
                    # print(f"Frame {frame_count}: Kasalar güncellendi.")
                else:
                    kasa_dict = {}  # Kasa yoksa boşalt
                    # print(f"Frame {frame_count}: Kasa bulunamadı.")

            # Güncel kasa merkezlerinin ortalamasını al
            if len(kasa_dict) > 0:
                # kasa_dict içindeki box'lardan merkezleri tekrar çıkar
                current_kasa_centers_for_avg = [
                    np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    for box in kasa_dict.values()
                ]
                current_kasa_avg_center = np.mean(current_kasa_centers_for_avg, axis=0)

                # ROI kalıbını kaydır
                # İlk kasa merkez ortalaması ile güncel kasa merkez ortalaması arasındaki farkı bul
                offset_vector = current_kasa_avg_center - initial_kasa_avg_center

                # Sabitlenmiş ROI kalıbını bu fark kadar kaydır
                current_roi_polygon = initial_roi_polygon_template + offset_vector.astype(np.int32)

                # print(f"Frame {frame_count}: ROI konumu güncellendi (kaydırıldı).")
            else:
                current_roi_polygon = None  # Kasa yoksa ROI gösterilemez
                print(f"Frame {frame_count}: Güncel kasa bulunamadığı için ROI gösterilemiyor.")
        else:
            current_roi_polygon = None  # Sabit ROI kalıbı yoksa gösterilemez
            cv2.putText(frame, "Sabit ROI kalibi bekleniyor...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Eğer bir güncel ROI varsa işleme devam et
        if current_roi_polygon is not None:
            # Balıkları takip et
            results = model.track(frame, persist=True)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []
            confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []

            active_ids = set()

            #for box, obj_id in zip(boxes, ids):
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
                    missing_counter[obj_id] = 0  # Göründü, sıfırla
                    last_seen_positions[obj_id] = (cx, cy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    """cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"""
                    cv2.putText(frame, f"ID:{obj_id} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Kayıp ID'leri güncelle
            for obj_id in list(last_seen_positions.keys()):
                if obj_id not in active_ids:
                    missing_counter[obj_id] = missing_counter.get(obj_id, 0) + 1

            # Takipten çıkanlar için kasaya sayım
            for obj_id in list(last_seen_positions.keys()):
                if (obj_id in counted_ids and
                        missing_counter.get(obj_id, 0) >= MISSING_THRESHOLD and
                        obj_id not in dropped_ids):

                    cx, cy = last_seen_positions[obj_id]

                    # Hangi kasaya düştüğünü kontrol et (güncel kasa konumları kullanılarak)
                    assigned_to_kasa = False
                    for kasa_id, box in kasa_dict.items():
                        if point_in_box(cx, cy, box):
                            kasa_counts[kasa_id] += 1
                            print(
                                f"Balık {obj_id} (kayıp) -> Kasa {kasa_id}. Total Fish Container {kasa_id}: {kasa_counts[kasa_id]}")
                            assigned_to_kasa = True
                            break

                    if assigned_to_kasa:
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                        dropped_ids.add(obj_id)
                    else:
                        dropped_ids.add(obj_id)

            # Güncel ROI kutusunu çiz
            cv2.polylines(frame, [current_roi_polygon], True, (255, 0, 255), 2)

            # Toplam balık sayısını göster
            cv2.putText(frame, f"Total Fish Count: {len(counted_ids)}",
                        (width - 300, 30),  # Sağ üst köşe
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "ROI gosterilemiyor...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Kasa kutularını çiz
        for _, box in kasa_dict.items():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Kasa sayaçlarını sol üst köşede sırayla yaz
        base_y = 40
        for kasa_id in sorted(kasa_counts.keys()):
            text = f"Kasa {kasa_id}: {kasa_counts[kasa_id]}"
            cv2.putText(frame, text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
            base_y += 30

        cv2.imshow("Fish Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()