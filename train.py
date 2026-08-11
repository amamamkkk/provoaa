import argparse
from pathlib import Path
from ultralytics import YOLO


DATASET_YAML = Path("datasets/weapon_dataset/data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение YOLO для детекции оружия")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help="Путь к data.yaml датасета")
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                        help="Базовая модель (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0",
                        help="0 = GPU, 'cpu' = только CPU")
    parser.add_argument("--bench", type=int, default=1,
                        help="1=режим замера (1 эпоха), 0=полное обучение")
    args = parser.parse_args()

    model = YOLO(args.model)

    epochs = 1 if args.bench else args.epochs
    results = model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        workers=0,
        save=True,
        project="runs/detect",
        name="weapon",
        verbose=True,
    )

    best = Path("runs/detect/weapon_train/weights/best.pt")
    out = Path("weights/weapon_detection.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    best.rename(out)
    print(f"\nГотовая модель сохранена: {out}")


if __name__ == "__main__":
    main()
