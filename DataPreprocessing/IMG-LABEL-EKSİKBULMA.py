"""import os


def rename_images_sequentially(folder_path, base_name="frame_svo2_", start_index=0, digits=4):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # Klasördeki tüm dosyaları al
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sadece resim dosyalarını filtrele ve sırala (isteğe bağlı ama iyi bir pratik)
    image_files = sorted([f for f in files if f.lower().endswith(image_extensions)])

    if not image_files:
        print(f"'{folder_path}' klasöründe desteklenen resim dosyası bulunamadı.")
        return

    print(f"'{folder_path}' klasöründeki resimler yeniden adlandırılıyor...")

    for i, old_name in enumerate(image_files):
        # Yeni dosya adını oluştur
        new_index = start_index + i
        new_name = f"{base_name}{new_index:0{digits}d}{os.path.splitext(old_name)[1]}"

        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"Yeniden adlandırıldı: '{old_name}' -> '{new_name}'")
        except Exception as e:
            print(f"Hata oluştu: '{old_name}' yeniden adlandırılamadı. Hata: {e}")

    print("Tüm resimler başarıyla yeniden adlandırıldı!")


# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    # Resimlerinizin bulunduğu klasör yolunu buraya yazın
    # ÖNEMLİ: Kendi klasör yolunuzla değiştirmeyi unutmayın!
    folder_to_rename = "bak/images"

    # Örnek kullanım:
    rename_images_sequentially(folder_to_rename, base_name="frame_svo2_", start_index=0, digits=4)"""

import os


def find_mismatched_files(images_folder, labels_folder):
    """
    Belirtilen iki klasörü karşılaştırır ve dosya uzantılarını dikkate almadan,
    birinde olup diğerinde olmayan dosyaları bulur.

    Args:
        images_folder (str): Resimlerin bulunduğu klasörün yolu (örn: "bak/images").
        labels_folder (str): Etiketlerin bulunduğu klasörün yolu (örn: "bak/labels").
    """
    print(f"'{images_folder}' ve '{labels_folder}' klasörlerindeki dosyalar karşılaştırılıyor...")

    # Resim klasöründeki tüm dosya adlarını (uzantısız) al
    image_files_base = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if
                        os.path.isfile(os.path.join(images_folder, f))}

    # Etiket klasöründeki tüm dosya adlarını (uzantısız) al
    label_files_base = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if
                        os.path.isfile(os.path.join(labels_folder, f))}

    # Resimlerde olup etiketlerde olmayan dosyaları bul
    in_images_not_in_labels = image_files_base - label_files_base

    # Etiketlerde olup resimlerde olmayan dosyaları bul
    in_labels_not_in_images = label_files_base - image_files_base

    if in_images_not_in_labels:
        print(f"\n--- '{images_folder}' klasöründe olup '{labels_folder}' klasöründe olmayan dosyalar (uzantısız): ---")
        for file_name in sorted(list(in_images_not_in_labels)):
            print(f"- {file_name}")
    else:
        print(f"\n'{images_folder}' klasöründe olup '{labels_folder}' klasöründe olmayan dosya bulunamadı.")

    if in_labels_not_in_images:
        print(f"\n--- '{labels_folder}' klasöründe olup '{images_folder}' klasöründe olmayan dosyalar (uzantısız): ---")
        for file_name in sorted(list(in_labels_not_in_images)):
            print(f"- {file_name}")
    else:
        print(f"\n'{labels_folder}' klasöründe olup '{images_folder}' klasöründe olmayan dosya bulunamadı.")


# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    # Kendi klasör yollarını buraya yaz!
    images_folder_path = "bak/images"
    labels_folder_path = "bak/labels"

    # Klasörlerin mevcut olduğundan emin ol (eğer yoksa hata verir)
    if not os.path.isdir(images_folder_path):
        print(f"Hata: '{images_folder_path}' klasörü bulunamadı.")
    if not os.path.isdir(labels_folder_path):
        print(f"Hata: '{labels_folder_path}' klasörü bulunamadı.")

    if os.path.isdir(images_folder_path) and os.path.isdir(labels_folder_path):
        find_mismatched_files(images_folder_path, labels_folder_path)
    else:
        print("Lütfen belirtilen klasör yollarını kontrol edin.")