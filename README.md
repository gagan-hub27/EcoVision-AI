# EcoVision AI
Python + Streamlit + YOLO + MongoDB Atlas waste recognition and management minor project.

## Scope and limitation
No model recognises literally every waste item. This starter supports expandable categories and routes unmapped model classes to General / unknown. It does not include trained weights or claim an accuracy result. Without `models/best.pt`, manual review mode tests the interface/database but is NOT AI recognition.

## Setup (Windows)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Copy `.env.example` to `.env`. In MongoDB Atlas create a cluster and database user, add your current IP to Network Access, copy the Python driver connection string, and replace the placeholders in `MONGODB_URI`. URL-encode special characters in the password. Never upload `.env` to GitHub.

Run:
```powershell
streamlit run app.py
```
MongoDB creates the `detections` collection automatically on first save. The app can open without Atlas, but database history and analytics require a successful connection.

## Dataset recommendation
Start with **TACO (Trash Annotations in Context)** for object detection research: real-world litter images with bounding-box annotations. It has uneven categories and does not represent every waste class. Inspect its current licence, annotations and class distribution before use.
For a simpler first classification experiment, TrashNet has six classes (cardboard, glass, metal, paper, plastic, trash); it is not a multi-object detection dataset and does not include every class in this app.

## YOLO dataset layout
```text
waste_dataset/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.yaml
```
Each label line is `class_id x_center y_center width height`, with normalised coordinates from 0 to 1. Example:
```yaml
path: C:/datasets/waste_dataset
train: images/train
val: images/val
test: images/test
names:
  0: plastic
  1: paper
  2: cardboard
  3: glass
  4: metal
```
Only list classes actually annotated in your dataset. Keep train/validation/test images separate by original image source. Inspect a sample of bounding boxes manually.

Train:
```powershell
yolo detect train model=yolov8n.pt data="C:/datasets/waste_dataset/data.yaml" epochs=50 imgsz=640 batch=8
```
Copy `runs/detect/train/weights/best.pt` to `models/best.pt`, then restart Streamlit. Update aliases in `modules/waste_guide.py` if your model's class names differ.

## Report evaluation
Measure per-class precision, recall and F1; detection mAP50 and mAP50-95; inference time; unknown/low-confidence rate; and manual review agreement. Use a held-out test set. Do not report training metrics as test accuracy.

## Safety
Do not use AI predictions to handle chemicals, medical waste, sharp objects, leaking containers or damaged batteries. Follow authorised local disposal rules. This is a decision-support student project, not a hazardous-waste detector.


## Deep learning pipeline
This project uses a CNN-based YOLO object detector with transfer learning. See `DEEP_LEARNING_GUIDE.md` and run `python training/train_yolo.py --data PATH_TO_DATA_YAML`. Evaluate on the held-out test split with `python training/evaluate_model.py --data PATH_TO_DATA_YAML`. Model weights are not bundled.
