# Emotion Detection using YOLO11

## Overview

This project implements a real-time facial emotion detection system using the YOLO11 object detection model. The model is trained on a facial emotion dataset downloaded from Roboflow and can detect human faces along with their corresponding emotional expressions.

The project demonstrates the complete deep learning pipeline, including dataset preparation, model training, and evaluation using Ultralytics YOLO.

---

## Features

- Facial emotion detection using YOLO11
- Roboflow dataset integration
- Automatic dataset download
- Model training using Ultralytics
- Performance evaluation
- Confusion matrix visualization
- Label distribution visualization

---

## Technologies Used

- Python
- YOLO11 (Ultralytics)
- Roboflow
- OpenCV
- PyTorch
- NumPy
- Matplotlib
- Google Colab

---

## Dataset

The dataset is downloaded directly from Roboflow.

Dataset Details:

- Source: Roboflow Universe
- Task: Object Detection
- Classes: Facial Emotions
- Format: YOLO

Download command:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("emotion-detection-5nu8v").project("facial-emotion-trpg7-07png")
version = project.version(1)
dataset = version.download("yolov11")
```

---


Install dependencies:

```bash
pip install ultralytics
pip install roboflow
pip install opencv-python
pip install torch torchvision
```


---

## Model Training

Training is performed using the Ultralytics YOLO framework.

```python
!yolo task=detect mode=train \
data={dataset.location}/data.yaml \
model='yolo11n.pt' \
epochs=50 \
imgsz=640
```

Training Parameters

| Parameter | Value |
|-----------|-------|
| Model | YOLO11 Nano |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Task | Object Detection |

---

## Results

After training, the following evaluation outputs are generated:

- Confusion Matrix
- Label Distribution
- Training Metrics
- Precision
- Recall
- mAP

Example:

```python
Image("/content/runs/detect/train/confusion_matrix.png", width=600)
```

```python
Image("/content/runs/detect/train/labels.jpg", width=600)
```

---

## Supported Emotions

Depending on the dataset, the model can detect emotions such as:

- Happy
- Sad
- Angry
- Fear
- Surprise
- Neutral
- Disgust

---

## Requirements

```
Python >= 3.10
ultralytics
roboflow
torch
torchvision
opencv-python
numpy
matplotlib
```

Install everything using:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Real-time webcam emotion detection
- Video emotion analysis
- Emotion tracking
- Face recognition integration
- Deploy using Flask or Streamlit
- Mobile deployment

---


