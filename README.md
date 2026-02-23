# Plant-Object-Detection-using-YOLOv8
This project demonstrates an end-to-end computer vision pipeline for detecting and classifying plant categories using YOLOv8.

The model was trained from scratch on a custom dataset consisting of:

- 🌿 Leafy plants  
- 🌵 Cactus  
- 🌱 Succulents  

The goal was not only to train a model, but to understand its behavior, limitations, and generalization performance.

---

## Problem Statement

Train a custom object detection model capable of detecting and classifying plant types in real-world images.

Unlike pre-trained datasets, this project required:

- Collecting custom images
- Manual bounding box annotation
- YOLOv8 training
- Evaluation on unseen plant species

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

### Training Command

```bash
pip install ultralytics
yolo detect train data=data.yaml model=yolov8n.pt epochs=30 imgsz=512
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
yolo detect predict \
model=runs/detect/train/weights/best.pt \
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
