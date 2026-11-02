"""Transfer-learning training pipeline for EcoVision AI waste detection.

Run from project root:
python training/train_yolo.py --data C:/datasets/waste_dataset/data.yaml --epochs 60
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained Ultralytics detection checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="For example 0 for GPU or cpu")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    # Transfer learning: start from pretrained weights and fine-tune on waste classes.
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        patience=15,
        optimizer="auto",
        pretrained=True,
        cos_lr=True,
        close_mosaic=10,
        degrees=8.0,
        translate=0.10,
        scale=0.40,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        project="runs/ecovision",
        name="waste_detection",
        plots=True,
        save=True
    )
    print("Training complete. Review validation metrics and plots.")
    print("Best checkpoint is normally at runs/ecovision/waste_detection/weights/best.pt")
    print("Copy that file to models/best.pt after checking class names.")
    print(results)

if __name__ == "__main__":
    main()
