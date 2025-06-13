from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")
    model.train(
        data="E:/halit/kasa-data.yaml",
        epochs=50,
        imgsz=720,
        batch=16,
        name="kasa_model",
        patience=4,
        device=0,
        dropout = 0.15,
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()



