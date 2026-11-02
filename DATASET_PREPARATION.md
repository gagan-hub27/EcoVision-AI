# Dataset preparation checklist

1. Choose an object-detection dataset such as TACO and review its licence and category annotations.
2. Decide which of the app's categories genuinely exist in the chosen dataset. Start with the categories with enough examples; do not fabricate labels.
3. Convert annotations to YOLO format if they are not already YOLO text files.
4. Split by source image into train, validation and test (for example 70/20/10); avoid near-duplicate images crossing splits.
5. Inspect images with drawn boxes and verify class IDs.
6. Check class imbalance and collect more examples for underrepresented classes.
7. Train a small YOLO model, evaluate on held-out test images, and document errors.
8. Copy the best weights to `models/best.pt`.
9. Match model class names to `ALIASES` in `modules/waste_guide.py`.
10. Test confidence thresholds and unknown-class behaviour before presenting.

TACO is suitable for a litter detection prototype but does not guarantee coverage of batteries, e-waste, chemicals or every household waste type. For those categories, source a separately licensed and accurately labelled dataset and evaluate them independently.
