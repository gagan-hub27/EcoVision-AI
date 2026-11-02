# Deep Learning in EcoVision AI

EcoVision AI uses **deep-learning object detection**, not only traditional image processing.

## 1. Model concept
The recommended pipeline fine-tunes a pretrained YOLO object-detection network on labelled waste images.

- **Convolutional Neural Network (CNN):** learns visual features such as edges, textures, shapes and object parts.
- **Backbone:** extracts feature maps from the image.
- **Neck / feature fusion:** combines information at different spatial scales so small and large waste objects can be detected.
- **Detection head:** predicts bounding boxes, object confidence and class probabilities.
- **Transfer learning:** starts from weights learned on a large image dataset, then adapts them to the waste dataset.
- **Data augmentation:** mosaic, horizontal flip, translation, scaling and small rotation help the model generalise.
- **Optimisation and loss:** the training framework optimises box localisation, classification and objectness-related detection losses using backpropagation and gradient-based optimisation.

Exact internal layer names and loss terms depend on the selected YOLO model/version.

## 2. Training workflow
1. Collect and label images with bounding boxes.
2. Split data into train, validation and test sets by source image.
3. Train using `training/train_yolo.py`.
4. Use validation metrics to select/checkpoint the model.
5. Evaluate only once the model is finalised on the held-out test set.
6. Copy the reviewed best checkpoint to `models/best.pt`.
7. Confirm class-name order and map class names in `modules/waste_guide.py`.

Example:
```powershell
python training/train_yolo.py --data "C:/datasets/waste_dataset/data.yaml" --epochs 60 --imgsz 640 --batch 8
```

Evaluation:
```powershell
python training/evaluate_model.py --weights models/best.pt --data "C:/datasets/waste_dataset/data.yaml"
```

## 3. Deep-learning concepts to explain in your viva
- **CNN:** a neural network designed to learn spatial features from images using convolution operations.
- **Convolution:** applies learned filters to extract local patterns.
- **Pooling / downsampling:** reduces spatial dimensions while retaining useful feature information.
- **Activation function:** introduces non-linearity so the network can learn complex patterns.
- **Transfer learning:** reuses pretrained visual features and fine-tunes them for waste categories.
- **Backpropagation:** computes gradients of the loss with respect to model parameters.
- **Epoch:** one pass through the training dataset.
- **Batch size:** number of training examples processed before an optimiser update.
- **Learning rate:** controls the size of parameter updates.
- **Overfitting:** model performs well on training images but poorly on unseen images.
- **Data augmentation:** creates varied training examples from labelled images.
- **Object detection:** predicts both class labels and object locations, unlike image classification which predicts a label for the whole image.

## 4. Metrics
For object detection, report:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- mAP@0.5 and mAP@0.5:0.95
- Inference time per image
- Per-class results and confusion matrix where available

Use a held-out test split. Do not claim accuracy based only on training output. Record the model name, dataset version, split sizes, image size, epochs, hardware and date for reproducibility.

## 5. Limitations
A model can only recognise classes represented sufficiently well in its training data. TACO and TrashNet have different task formats and limited category coverage. Unknown, low-confidence or hazardous-looking items should be routed to human review. A visual prediction cannot establish chemical composition, contamination or battery safety.
