# 🌿 Plant Object Detection using YOLOv8
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![Roboflow](https://img.shields.io/badge/Roboflow-Annotation-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

This project demonstrates an end-to-end computer vision pipeline for detecting and classifying plant categories using YOLOv8.

The model was trained from scratch on a custom dataset consisting of:

- 🌿 Leafy plants  
- 🌵 Cactus  
- 🌱 Succulents  

The objective was to build a custom object detection model, analyze its behavior, evaluate its performance, and understand its limitations and generalization capabilities.

---

## Problem Statement

Train a custom object detection model capable of detecting and classifying plant types in real-world images.

Unlike using large pre-trained datasets, this project required:

- Collecting a custom image dataset  
- Manually annotating bounding boxes  
- Training a YOLOv8 model  
- Evaluating performance on unseen plant images  

---

## Dataset

Total images:

- Leafy plants: 45
- Cactus: 24
- Succulents: 24

Images were captured under varying:

- Lighting conditions
- Backgrounds
- Angles
- Distances

Dataset was split into:
- 70% Training
- 20% Validation
- 10% Test

---

## Annotation

Images were manually annotated using Roboflow.

The dataset was exported in YOLOv8 format and resized to 512x512 resolution.

---

## Model Training

Model: `yolov8n.pt`  
Epochs: 30  
Image Size: 512  

Training was performed locally using the Ultralytics framework.

### Build environment
Environment is built using uv package manager. Dependencies are in `pyproject.toml`. 

```bash
uv sync
```

### Training Command

```bash
uv run yolo detect train data=data.yaml model=yolov8n.pt epochs=30 imgsz=512
```

---

## Evaluation Metrics

After 30 epochs of training, the model achieved:

- **mAP@50:** 0.92  
- **mAP@50-95:** 0.65  
- **Precision:** ~0.90  
- **Recall:** ~0.90  

### What does this mean?

- The model correctly detects plant categories in most cases.
- Bounding box localization is strong at moderate overlap thresholds (50% IoU).
- Performance decreases at stricter overlap thresholds (95% IoU), which is expected with a small custom dataset.
- Leafy plants performed best due to higher representation in the training data.

Confusion matrix and training curves can be found in the `/results` folder.

---

## Generalization Testing (Unseen Images)

To evaluate real-world robustness, the model was tested on **28 completely new plant images**:

- 11 leafy plants  
- 9 cactus  
- 8 succulents  

### Inference Command

```bash
uv run yolo detect predict \
model=weights/best.pt \
source=test_images \
conf=0.25
```

## Results Summary

The model was evaluated on 28 unseen plant images:

- 11 leafy plants  
- 9 cactus  
- 8 succulents  

### Prediction Breakdown

- 3 images were not detected  
  - 2 leafy  
  - 1 cactus  

- 2 cactus misclassified as leafy  
- 1 leafy misclassified as cactus  
- 1 succulent misclassified as leafy  

The remaining images were correctly classified.

Overall, the model demonstrated strong performance on familiar visual patterns and moderate degradation when tested on visually atypical samples.

---
## Real-Time Deployment (Webcam Testing)

After training the YOLOv8 model, I connected it to my laptop’s webcam to test real-time detection.

```bash
uv run yolo detect predict model=weights/best.pt source=0 show=True conf=0.4
```

The model runs at ~20–30ms per frame and successfully detects plants in live video.

### 🌵 Cactus (very confident)
<img src="screenshots/cactus_detection.png" width="500">

### 🍃 Leafy Detection
<img src="screenshots/leafy_detection.png" width="500">

### 🌿 Succulent (also strong)
<img src="screenshots/succulent_detection.png" width="500">

### 😄 Fun Observation
At one point, the model detected **me** as a leafy plant.  
This likely happened due to similar color patterns and limited dataset size — a great reminder of how models generalize (or overfit).

This step completed the full ML lifecycle:
data collection → annotation → training → evaluation → real-time deployment.

---

## Observations

- Cactus samples with smoother or denser structure were more likely to be confused with leafy plants.
- A purple inchplant introduced color variation not present in training data, affecting classification.
- Succulent misclassifications suggest sensitivity to background and lighting conditions.
- Leafy plants performed best due to higher representation in the training dataset.
- The model responds to statistical visual patterns (texture, edges, density) rather than semantic plant categories.

---

## Key Insights

- Dataset size and class balance significantly influence performance.
- Visual diversity improves generalization more effectively than model scaling.
- Validation accuracy does not guarantee robustness on unseen data.
- Failure cases provide more learning value than perfect predictions.
- Small datasets can produce promising results but remain fragile under distribution shifts.

---

## Future Improvements

To improve performance and robustness:

1. Expand dataset (especially cactus & succulents)
2. Improve class balance
3. Increase visual diversity (lighting, angles, backgrounds)
4. Apply controlled augmentation
5. Compare retrained model versions
6. Experiment with larger YOLO architectures after dataset expansion

More diverse and balanced data → stronger generalization.
