import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO

# --- Model ve Video Yolları (Değişmedi) ---
MODEL_PATH = "models/best.pt"
KASA_MODEL_PATH = "models/best-kasa.pt"
# VIDEO_PATH artık kullanılmayacak, yerine SVO_PATH kullanılacak
SVO_PATH = "videos/2/video.svo2"  # Burası SVO2 dosyanızın yolu olmalı



# --- ROI Seçim Fonksiyonu (Değişmedi) ---
def select_roi(window_name: str, frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Kullanıcının verilen kare üzerinde bir ROI seçmesini sağlar.
    Seçilen ROI'nin koordinatlarını (x, y, w, h) döndürür.
    Geçersiz bir seçim yapılırsa RuntimeError fırlatır.
    """
    print(f"'{window_name}' penceresinde ROI seçmek için fareyi kullanın ve bitince ENTER veya SPACE tuşuna basın.")
    print("Seçimi iptal etmek için 'c' tuşuna basın.")

    roi = cv2.selectROI(window_name, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = map(int, roi)

    if w == 0 or h == 0:
        raise RuntimeError("Geçerli bir ROI seçilmedi. Genişlik ve yükseklik sıfır olamaz.")
    return x, y, w, h


# --- Noktanın Kutu İçinde Olup Olmadığını Kontrol Eden Fonksiyon (Değişmedi) ---
def point_in_box(cx, cy, box):
    x1, y1, x2, y2 = map(int, box)
    return x1 <= cx <= x2 and y1 <= cy <= y2


# --- Ana İşleme Fonksiyonu (SVO2 Entegrasyonu Yapıldı) ---
def main():
    # --- Modelleri Yükle (Değişmedi) ---
    model = YOLO(MODEL_PATH)
    kasa_model = YOLO(KASA_MODEL_PATH)
    print(f"Balık modeli '{MODEL_PATH}' ve Kasa modeli '{KASA_MODEL_PATH}' yüklendi.")

    # --- ZED Kamerayı (SVO dosyasını) Başlatma ---
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(SVO_PATH)
    init_params.svo_real_time_mode = False  # Gerçek zamanlı modu kapat
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # GPU uyumluluğu ve performans için

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED kamerası açılamadı veya SVO dosyası okunamadı: {repr(err)}")
        return

    # SVO dosyasından FPS'i al
    camera_info = zed.get_camera_information()
    fps = camera_info.camera_configuration.fps
    print(f"SVO dosyası FPS: {fps}")

    # Videoyu belirli bir saniyeden başlat (MP4 kodundaki start_frame mantığı)
    start_second = 50  # MP4 kodundaki gibi 35. saniyeden başlat
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

    # --- Değişkenlerin Başlangıç Değerleri (MP4 kodundaki gibi) ---
    roi = None
    counted_ids = set()
    last_seen_positions = {}
    dropped_ids = set()
    kasa_counts = {}
    missing_counter = {}  # 🔁 ID: kaç frame'dir kayıp
    MISSING_THRESHOLD = 10  # 🔢 Kaç frame boyunca yoksa artık düştü kabul edelim
    frame_count = 0
    kasa_dict = {}  # Başta tanımla ki her yerde erişilebilsin

    # SVO dosyasından ilk kareyi al ve ROI seçimi için kullan
    initial_frame = None
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        initial_frame = image_zed.get_data()
        if initial_frame.shape[2] == 4:  # Eğer RGBA/BGRA ise, Yani ALFA DEPTH  kanalı varsa:
            initial_frame = cv2.cvtColor(initial_frame, cv2.COLOR_RGBA2RGB)
    else:
        print("SVO dosyasından ilk kare alınamadı.")
        zed.close()
        return

    try:
        initial_frame_for_roi = initial_frame.copy()
        roi = select_roi(f"Balıklar için ROI Seçimi - {SVO_PATH.split('/')[-1]}",initial_frame_for_roi)
        print(f"Seçilen Balık ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    except RuntimeError as e:
        print(e)
        zed.close()
        return

    # --- Ana Döngü: Kareleri Oku ve İşle ---
    while True:
        # ZED SVO'dan kare yakalama
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            current_frame_ocv = image_zed.get_data()
            if current_frame_ocv.shape[2] == 4:
                # BÖYLEYDİ. current_frame_ocv = cv2.cvtColor(current_frame_ocv, cv2.COLOR_BGRA2RGB)
                current_frame_ocv = cv2.cvtColor(current_frame_ocv, cv2.COLOR_BGRA2RGB)
                #  current_frame_ocv = cv2.cvtColor(current_frame_ocv, cv2.COLOR_BGRA2BGR)
            #image_for_yolo = current_frame_ocv
            image_for_yolo = current_frame_ocv.copy()

            frame_count += 1
            display_frame = cv2.cvtColor(current_frame_ocv, cv2.COLOR_RGB2BGR)
            #display_frame = current_frame_ocv.copy()

            key = cv2.waitKey(1) & 0xFF

            # ROI sıfırlama tuşu (değişmedi)
            if key == ord('r'):
                try:
                    roi = select_roi("ROI Seç", display_frame)  # display_frame kullanıldı
                    counted_ids.clear()
                    dropped_ids.clear()
                    last_seen_positions.clear()
                    print("Yeni ROI:", roi)
                except RuntimeError as e:
                    print(e)
                    roi = None

            if roi is not None:
                x, y, w, h = roi

                # --- Kasaları tespit ve sırala (MP4 koduyla aynı mantık) ---
                if frame_count == 1 or (frame_count % 100 == 0):  # İlk karede veya her 100 frame'de bir
                    # YOLO modeline RGB formatında gönder
                    kasa_results = kasa_model(image_for_yolo, conf=0.9, verbose=False)[0]  # track yerine direkt model çağrımı
                    kasa_boxes = []
                    # Sadece "fish_box_custom" sınıfından olan kasaları al
                    for box_r in kasa_results.boxes:
                        detected_class_name = kasa_model.names[int(box_r.cls.cpu().numpy()[0])]
                        if detected_class_name == "fish_box_custom":  # Hedef kasa sınıfı
                            kasa_boxes.append(box_r.xyxy.cpu().numpy()[0])

                    if len(kasa_boxes) > 0:
                        kasa_boxes_sorted = sorted(kasa_boxes, key=lambda b: b[0])
                        kasa_dict = {i: box for i, box in enumerate(kasa_boxes_sorted)}
                    else:
                        kasa_dict = {}  # Kasa yoksa boş sözlük

                # Kasa sayımlarını sıfırla veya güncelle (mevcut kasalar için)
                temp_kasa_counts = {kasa_id: 0 for kasa_id in kasa_dict.keys()}
                for kasa_id, count in kasa_counts.items():
                    if kasa_id in temp_kasa_counts:
                        temp_kasa_counts[kasa_id] = count  # Eski sayımları koru
                kasa_counts = temp_kasa_counts

                # --- Balıkları takip et (MP4 koduyla aynı mantık) ---
                results = model.track(image_for_yolo, conf=0.35, persist=True, verbose=False)[0]  # YOLO modeline RGB formatında gönder

                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []
                # Confidence değerlerini al
                confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []


                active_ids = set()

                for box, obj_id, conf in zip(boxes, ids, confs): # conf'u zip'e ekle
                    obj_id = int(obj_id)
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    active_ids.add(obj_id)

                    # ROI içinde mi?
                    if x <= cx <= x + w and y <= cy <= y + h:
                        if obj_id not in counted_ids:
                            counted_ids.add(obj_id)

                    if obj_id in counted_ids:
                        missing_counter[obj_id] = 0  # 👈 Göründü, sıfırla
                        last_seen_positions[obj_id] = (cx, cy)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Confidence değerini de etiketle
                        label_text = f"ID:{obj_id} Conf:{conf:.2f}"
                        cv2.putText(display_frame, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 🔴 KAYIP ID'LERİ GÜNCELLE ve Takipten çıkanları kasaya say
                for obj_id in list(last_seen_positions.keys()):
                    if obj_id not in active_ids:
                        missing_counter[obj_id] = missing_counter.get(obj_id, 0) + 1

                    if missing_counter.get(obj_id, 0) >= MISSING_THRESHOLD and obj_id not in dropped_ids:
                        cx, cy = last_seen_positions[obj_id]

                        # Hangi kasaya düştüğünü kontrol et
                        assigned_to_any_box = False
                        for kasa_id, box in kasa_dict.items():
                            if point_in_box(cx, cy, box):
                                kasa_counts[kasa_id] = kasa_counts.get(kasa_id,
                                                                       0) + 1  # Mevcut veya yeni kasa için sayım
                                print(f"Balık {obj_id} -> Kasa {kasa_id}")
                                assigned_to_any_box = True
                                break  # Tek bir kasaya düşebilir

                        # Eğer hiçbir kasaya düşmediyse ve hala takipte değilse, dropped_ids'e ekle
                        if assigned_to_any_box:
                            cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)  # Kırmızı nokta
                            dropped_ids.add(obj_id)
                        elif obj_id not in dropped_ids:  # Eğer düşmediyse ama hala kayıpsa
                            pass

                # --- Görüntüye Çizimler ---
                # ROI kutusu
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(display_frame, f"Toplam Sayilan Balik: {len(counted_ids)}",
                            (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Kasa kutularını çiz (yazı yazma)
                for kasa_id, box in kasa_dict.items():
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    # Kasa ID'sini ve sayısını kasanın üstüne yaz
                    kasa_label = f"Kasa {kasa_id}"
                    cv2.putText(display_frame, kasa_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # 🟨 Kasa sayaçlarını sol üst köşede sırayla yaz
                base_y = 40
                for kasa_id in sorted(kasa_counts.keys()):
                    text = f"Kasa {kasa_id}: {kasa_counts[kasa_id]}"
                    cv2.putText(display_frame, text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2)
                    base_y += 30  # alt satıra geç

            cv2.imshow(f"ZED SVO Fish Counter - {SVO_PATH.split('/')[-1]}", display_frame)

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