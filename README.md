# Hand Gesture Recognition Using MediaPipe & TensorFlow Lite

[[English](README.md) / [Japanese](README_JA.md)]

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8 - 3.12](https://img.shields.io/badge/Python-3.8%20--%203.12-brightgreen.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-orange.svg)](https://mediapipe.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-yellow.svg)](https://tensorflow.org/)

> **Note**
> <br>A model zoo repository collecting various keypoint classification models is available here:
> <br>→ [Kazuhito00/hand-keypoint-classification-model-zoo](https://github.com/Kazuhito00/hand-keypoint-classification-model-zoo)

---

## Overview

This project implements real-time **Hand Sign Recognition** (static poses) and **Finger Gesture Recognition** (dynamic motion tracking) using **Google MediaPipe** and lightweight **Multilayer Perceptron (MLP)** models converted to **TensorFlow Lite (TFLite)**.

It captures video from your webcam, tracks 21 3D hand landmarks in real-time, extracts normalized relative coordinates, and classifies:
1. **Static Hand Signs**: `Open`, `Close` (Fist), `Pointer` (Pointing finger), etc.
2. **Dynamic Finger Gestures**: `Stop`, `Clockwise`, `Counter Clockwise`, and `Move` tracked using index fingertip history over time.

![Demo Animation](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

---

## Repository Contents

* **Real-time Inference Application** (`app.py`): Live camera processing and dataset logging.
* **Pretrained Models** (`model/`): Quantized TFLite models for sub-millisecond edge CPU inference.
* **Datasets & Labels**: CSV keypoint datasets and class label dictionaries.
* **Model Training Notebooks** (`.ipynb`): Jupyter notebooks for custom training, hyperparameter tuning, confusion matrix evaluation, and TFLite export.
* **Project Breakdown Document** (`PROJECT_BREAKDOWN.md`): Comprehensive architecture, mathematics, data preprocessing, and engineering guide.

---

## Requirements & Dependencies

* **Python**: 3.8 to 3.12
* **OpenCV**: `opencv-python >= 4.6.0`
* **MediaPipe**: `mediapipe >= 0.8.4` (0.10+ on Python 3.12)
* **TensorFlow**: `tensorflow >= 2.9.0` (or `tensorflow-cpu` / `tflite-runtime`)
* **Scikit-learn**: `scikit-learn >= 1.0.2` (used during training & evaluation)
* **Matplotlib**: `matplotlib >= 3.5.1` (used for plotting confusion matrices)

---

## Quick Start & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/watermelon588/hand-gesture-recognition-using-mediapipe.git
cd hand-gesture-recognition-using-mediapipe
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### CLI Arguments & Options
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--device` | int | `0` | Camera device index |
| `--width` | int | `960` | Webcam capture frame width |
| `--height` | int | `540` | Webcam capture frame height |
| `--use_static_image_mode` | flag | `False` | Run full detection on every frame (MediaPipe static mode) |
| `--min_detection_confidence` | float | `0.7` | Hand landmark detection confidence threshold |
| `--min_tracking_confidence` | float | `0.5` | Landmark tracking confidence threshold |

Example with custom camera index and resolution:
```bash
python app.py --device 0 --width 1280 --height 720
```

---

## 🎭 Interactive Meme Gesture Controller

Control and showcase memes live on camera using hand signs! Mimic the pose from each meme reference, and the system locks in the matching meme onto an in-video card overlay with a glowing HUD and hold progress indicator.

| ID | Reference Meme | Target Gesture | Handedness | How to Perform |
| :---: | :---: | :--- | :---: | :--- |
| **1** | ![Cat Shaka](memes/1.jpg) | 🤙 **Cat Shaka** | **Left or Right** | Extend **Thumb & Pinky** outward, fold middle 3 fingers ("Call me" / Shaka). |
| **2** | ![Monkey Pointing](memes/2.jpg) | ☝️ **1 Finger: Monkey Pointing** | **Left or Right** | Point **Index finger straight UP** (1 finger count), fold other fingers into a fist. |
| **4** | ![Monkey Thumbs Up](memes/4.jpg) | 👍 **Monkey Thumbs Up** | **Left or Right** | Point **Thumb straight UP**, clench other 4 fingers in a fist. |
| **5** | ![Cat 4-Finger Hold](memes/5.png) | 🐾 **4 Fingers: Cat 4-Finger Hold** | **Left or Right** | Raise **Index, Middle, Ring & Pinky UP** (4 finger count), thumb tucked/folded across palm. |
| **7** | ![J. Cole Open](memes/7.jpg) | 🖐️ **5 Fingers: J. Cole Open Hand** | **Left or Right** | Open all **5 fingers flat out** facing the camera (5 finger count / "Stop / Chill" pose). |
| **8** | ![Cool Cat Peace](memes/8.png) | ✌️ **2 Fingers: Cool Cat Peace** | **Left or Right** | Extend **Index & Middle fingers in a Peace/V sign** (2 finger count), fold ring and pinky. |

### 🔄 Dynamic Hand Gestures (Start & Close / Stop)

| Dynamic Motion | Action | Description |
| :---: | :---: | :--- |
| **↺ Counter-Clockwise (CCW)** | **START Detection** | Draw a circle counter-clockwise in the air with your pointer finger to start meme detection. |
| **↻ Clockwise (CW)** | **STOP & CLOSE Meme** | Draw a circle clockwise in the air with your pointer finger to dismiss the active meme and pause detection. |

* **Keyboard Controls**: Press `S` to toggle detection on/off, `C` to close the active meme.

### Stability & Experience Features
* **Zero External Windows**: Memes render directly in the top-right corner of the video window inside a letterboxed frosted-glass card.
* **Hold-to-Confirm Smoothing**: Hold any gesture for ~0.3s (8 frames) to activate. Transient hand movements or passing poses never trigger random memes.
* **Live Progress Bar**: Displays `Holding: <Gesture> (XX%)` on the card before locking in.
* **Full Left & Right Hand Support**: Powered by hand-agnostic joint kinematic geometry and finger count angle tracking.
* **Dynamic Air Gesture Showcase**: Real-time trajectory modeling (`PointHistoryClassifier`) controls app lifecycle without touching the keyboard.

---

## Docker Execution

```bash
docker build -t hand_gesture .

# Run on Linux with X11 display:
xhost +local: && \
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  hand_gesture:latest

python app.py
```

---

## Directory Structure

```
.
├── app.py                                # Main application & interactive inference pipeline
├── keypoint_classification.ipynb         # Hand sign classifier training & TFLite export
├── point_history_classification.ipynb    # Finger motion gesture training & TFLite export
├── requirements.txt                      # Project dependency specification
├── Dockerfile                            # Docker container configuration
├── LICENSE                               # Apache 2.0 License
├── README.md                             # English Documentation
├── README_JA.md                          # Japanese Documentation
├── PROJECT_BREAKDOWN.md                  # Comprehensive technical deep dive & guide
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint.csv                  # Training dataset (Class ID + 42 normalized coords)
│   │   ├── keypoint_classifier.hdf5      # Saved Keras model
│   │   ├── keypoint_classifier.tflite    # Quantized TFLite inference model
│   │   ├── keypoint_classifier_label.csv # Class names (Open, Close, Pointer)
│   │   └── keypoint_classifier.py        # TFLite inference wrapper class
│   └── point_history_classifier/
│       ├── point_history.csv             # Training dataset (Class ID + 32 historical coords)
│       ├── point_history_classifier.hdf5 # Saved Keras model
│       ├── point_history_classifier.tflite # Quantized TFLite inference model
│       ├── point_history_classifier_label.csv # Class names (Stop, Clockwise, Counter Clockwise, Move)
│       └── point_history_classifier.py   # TFLite inference wrapper class
└── utils/
    ├── __init__.py
    └── cvfpscalc.py                      # FPS calculation utility with smoothing deque
```

---

## Interactive Data Logging & Custom Training

You can easily record custom hand signs and finger movements directly from the running application.

### Key Controls in `app.py`:
- `ESC`: Exit application
- `n`: Normal inference mode (no recording)
- `k`: Keypoint logging mode (`MODE: Logging Key Point`)
- `h`: Point history logging mode (`MODE: Logging Point History`)
- `0` - `9`: Record the current gesture under class ID `0` through `9` into the respective dataset CSV.

---

### 1. Training a Custom Hand Sign Model

1. Run `python app.py`.
2. Press `k` to enter keypoint logging mode.
3. Show your hand pose to the camera and press a digit key (`0`–`9`) repeatedly across different angles and distances to save samples to `model/keypoint_classifier/keypoint.csv`.
4. Open [keypoint_classification.ipynb](keypoint_classification.ipynb) in Jupyter Notebook.
5. Set `NUM_CLASSES` to match your number of classes.
6. Run all cells to train the model, evaluate confusion matrices, and export `keypoint_classifier.tflite`.
7. Update class names in `model/keypoint_classifier/keypoint_classifier_label.csv`.

#### Hand Sign Neural Network Architecture
* **Input Layer**: 42 features (21 landmarks $\times$ $(x, y)$ relative to wrist, normalized by max absolute value)
* **Dropout (0.2)**
* **Dense (20 units, ReLU)**
* **Dropout (0.4)**
* **Dense (10 units, ReLU)**
* **Dense (NUM_CLASSES, Softmax)**

---

### 2. Training a Custom Finger Gesture Model

1. Run `python app.py`.
2. Press `h` to enter point history logging mode.
3. Point your index finger (Pointer hand sign) and perform a dynamic motion (e.g. clockwise circle, swipe). Press a digit key (`0`–`9`) to log to `model/point_history_classifier/point_history.csv`.
4. Open [point_history_classification.ipynb](point_history_classification.ipynb) in Jupyter Notebook.
5. Set `NUM_CLASSES` to match your number of gesture classes.
6. Run all cells to train the MLP (or optional LSTM) model and export `point_history_classifier.tflite`.
7. Update class names in `model/point_history_classifier/point_history_classifier_label.csv`.

#### Finger Gesture Neural Network Architecture
* **Input Layer**: 32 features (16 frames $\times$ $(x, y)$ index fingertip relative coordinates)
* **Dropout (0.2)**
* **Dense (24 units, ReLU)**
* **Dropout (0.5)**
* **Dense (10 units, ReLU)**
* **Dense (NUM_CLASSES, Softmax)**
*(An alternative LSTM architecture with 16 recurrent units is also included in the notebook).*

---

## Application Examples & Community Use Cases

* [Control DJI Tello Drone with Hand Gestures](https://towardsdatascience.com/control-dji-tello-drone-with-hand-gestures-b76bd1d4644f)
* [Classifying American Sign Language Alphabets on the OAK-D](https://www.cortic.ca/post/classifying-american-sign-language-alphabets-on-the-oak-d)

---

## References & Acknowledgments

* [Google MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
* [Kazuhito00/mediapipe-python-sample](https://github.com/Kazuhito00/mediapipe-python-sample)
* Original Author: **Kazuhito Takahashi** ([@KzhtTkhs](https://twitter.com/KzhtTkhs))

---

## License

This project is open-source under the [Apache 2.0 License](LICENSE).
