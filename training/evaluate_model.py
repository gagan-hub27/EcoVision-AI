"""Evaluate a trained YOLO model on a held-out test split."""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="")
    args = p.parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        device=args.device or None,
        plots=True,
        save_json=True
    )
    print("Detection metrics:")
    print("mAP50:", float(metrics.box.map50))
    print("mAP50-95:", float(metrics.box.map))
    print("Precision:", float(metrics.box.mp))
    print("Recall:", float(metrics.box.mr))
    print("Per-class mAP50-95:", metrics.box.maps)

if __name__ == "__main__":
    main()
