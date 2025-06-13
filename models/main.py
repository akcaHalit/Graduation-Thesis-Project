# dropout + Augment --> Path'ler yanlış sanırım yanlış vermişim e: olarak vermemişim.
# Şuana kadarki en iyi modelim:
from torch.onnx.symbolic_opset12 import dropout
from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")  # veya kendi modelin
    model.train(
        data="E:/halit/data.yaml",
        epochs=50,
        imgsz=960,
        batch=4,
        name="yolo_custom_fish_gpu_11s",
        patience=4,
        augment=True,
        device=0,
        dropout = 0.07
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # (Windows için güvenli başlatma)
    main()

#AUGMENT yapmak lazım gibi ÇOK HIZLI ÖĞRENDİ.     0.856      0.905      0.934      0.677 NORMAL DEĞİL.