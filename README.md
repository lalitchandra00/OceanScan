# OceanScan

Object detection on **side-scan sonar imagery** using a custom-trained **YOLOv8n** model, with a noise-filtering preprocessing pipeline.

Sonar images (e.g. from Side Scan Sonar systems) are notoriously noisy. OceanScan first cleans each image (speckle / salt-and-pepper removal, edge-preserving smoothing, local contrast enhancement) and then detects objects with a lightweight YOLOv8n detector trained on a custom sonar dataset.

**Detected classes:**

| ID | Class    |
|----|----------|
| 0  | body     |
| 1  | plane    |
| 2  | ship     |
| 3  | target   |

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Workflow / Pipeline](#workflow--pipeline)
5. [Notebooks](#notebooks)
6. [Datasets](#datasets)
7. [Usage](#usage)
8. [Outputs](#outputs)
9. [Key Parameters](#key-parameters)
10. [Environment](#environment)

---

## Features

- **Noise filtering** preprocessing tuned for sonar imagery:
  - **Median filter** — removes speckle / salt-and-pepper noise (sonar-typical).
  - **Bilateral filter** — edge-preserving smoothing that keeps target edges sharp.
  - **Light CLAHE** — gentle local contrast (clip 1.2, 16x16 tiles) so flat seabed areas are not over-amplified.
- **Custom object detector** — YOLOv8n trained from scratch on the noise-filtered sonar dataset with early stopping.
- **Versatile inference** — run predictions on:
  - Static **images** (`input/image/`)
  - **Videos** (`input/video/`)
  - **Live webcam** frames (real-time)
- **Four-class detection** of `body`, `plane`, `ship`, and `target`.
- **Save-on-detect** — only frames with detections above a confidence threshold are written to disk (saves disk space on long runs).
- **Archiving** — raw and noise-filtered inputs are kept for provenance/review.
- Runs on **CPU** (no GPU required).

---

## Project Structure

```
OceanScan/
├── README.md
├── requirements.txt
├── .gitignore
│
├── backend/                     # Model, noise filtering & prediction notebooks
│   ├── best.pt                  # Trained model weights used for prediction
│   ├── model.ipynb              # Train YOLOv8n + evaluate + export best.pt
│   ├── noise_filtering.ipynb    # Builds preprocess_for_model + batch dataset prep
│   ├── image_prediction.ipynb   # Predictions on static images
│   ├── video_prediction.ipynb   # Predictions on videos
│   ├── realtime_prediction.ipynb# Live webcam predictions
│   └── runs/
│       └── dataset2_yolov8n/    # Training run artifacts (curves, confusion matrix, results.csv)
│
├── dataset2/                    # Raw source dataset
│   ├── train/                   #  402 images + labels
│   ├── valid/                   #  110 images + labels
│   └── text/                    #  test images only (46 images)
│
├── input/                       # Where you drop new media to predict
│   ├── image/                   # static images
│   ├── video/                   # videos (.mp4 / .avi / .mov / .mkv / .m4v)
│   └── realtime/                # archived webcam frames
│
└── output/                      # All generated results (regenerable, git-ignored)
    ├── noise_filter/            # noise-filtered datasets & inputs
    ├── predictions/             # annotated detections (image / video / real-time)
    └── test_predictions/        # dataset2 test-set predictions
```

---

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs:

- `numpy`, `opencv-python`, `matplotlib`, `pandas`, `tqdm`
- `ultralytics` (YOLOv8 / YOLO ecosystem, v8.4.131)
- `torch` (CPU build: 2.13.0+cpu — no GPU needed)

> The notebooks also auto-detect and install any missing dependencies on **Run All**, so they can be opened and executed directly even without a pre-built environment.

---

## Workflow / Pipeline

The recommended order of execution is:

```
noise_filtering.ipynb  →  model.ipynb  →  image_prediction.ipynb
                                        →  video_prediction.ipynb
                                        →  realtime_prediction.ipynb
```

### 1. Noise filtering (`noise_filtering.ipynb`)

Builds the reusable `filter_noise(image, ...)` and `preprocess_for_model(image, ...)` functions, then:

- Batch-processes the whole `dataset2/` folder into `output/noise_filter/train_noise_filter/` (with `train` and `valid` subsets).
- Uses the **same function at prediction time**: new images are noise-filtered **first**, then passed to the model.
- Copies YOLO labels unchanged — bounding boxes are normalized so they stay valid after denoising/resizing.

### 2. Training (`model.ipynb`)

Trains **YOLOv8n** on the noise-filtered dataset:

- Train set: `output/noise_filter/train_noise_filter/train` (402 images)
- Val set: `output/noise_filter/train_noise_filter/valid` (110 images)
- 4 classes: `body`, `plane`, `ship`, `target`
- `imgsz=640`, `batch=8`, `device=cpu`, `workers=0`
- `epochs=20`, `patience=15` (early stopping on val mAP)
- Saves the best weights to `backend/best.pt` for the prediction notebooks.
- Also plots training curves and evaluates mAP on the validation split.

### 3-5. Prediction (image / video / real-time)

Each prediction notebook runs **noise filtering → YOLOv8n inference → save-on-detect**.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `backend/noise_filtering.ipynb` | Defines the preprocessing pipeline and prepares the noise-filtered training dataset. |
| `backend/model.ipynb` | Trains YOLOv8n, plots curves, evaluates, exports `best.pt`. |
| `backend/image_prediction.ipynb` | Predicts on static images from `input/image/` and on the dataset2 test set. |
| `backend/video_prediction.ipynb` | Predicts on videos from `input/video/`, saving only the frames with detections. |
| `backend/realtime_prediction.ipynb` | Live webcam detection, saving snapshots of frames with detections. |

---

## Datasets

### Raw dataset (`dataset2/`)

| Split   | Images |
|---------|--------|
| train   | 402    |
| valid   | 110    |
| text    | 46     |

- `dataset2/train/` and `dataset2/valid/` contain `images/` and `labels/` (YOLO-format `.txt` labels).
- `dataset2/text/` contains the test images (no labels) used to produce `output/test_predictions/`.

### Noise-filtered dataset (`output/noise_filter/train_noise_filter/`)

The same train/valid splits after preprocessing, with a `data.yaml` describing the 4 classes. This is what the model is trained on.

---

## Usage

1. **Add your media** for inference:
   - Images → `input/image/`
   - Videos → `input/video/`
   - Webcam → use `realtime_prediction.ipynb`
2. **Open the relevant notebook** (image / video / real-time prediction).
3. **Run All** — missing dependencies are auto-installed.
4. Collect the annotated results from `output/`.

### Confidence thresholds

- Image prediction: `CONF_THRESHOLD = 0.25`
- Video / real-time prediction: `CONF_THRESHOLD = 0.50`

Boxes are only drawn / frames only saved when a detection exceeds the threshold.

---

## Outputs

| Output path | Description |
|-------------|-------------|
| `output/predictions/image_prediction/` | Annotated static-image predictions. |
| `output/predictions/video_prediction/` | Frames with detections from videos. |
| `output/predictions/realtime_prediction/image/` | Webcam snapshots with detections. |
| `output/test_predictions/` | Predictions over the dataset2 test set (46 images). |
| `output/noise_filter/train_noise_filter/` | Noise-filtered train/valid dataset for training. |
| `output/noise_filter/input_noise_filter/` | Noise-filtered copies of real-time/video/input media. |

> `output/` is git-ignored — it is regenerable from the notebooks.

---

## Key Parameters

| Parameter | Value | Where |
|-----------|-------|-------|
| Model | YOLOv8n (nano) | `model.ipynb` |
| Input size | `imgsz=640` | `model.ipynb` |
| Batch size | `batch=8` | `model.ipynb` |
| Epochs | `epochs=20` | `model.ipynb` |
| Early stopping patience | `patience=15` | `model.ipynb` |
| Device | `cpu` | `model.ipynb` |
| Image prediction conf. | `0.25` | `image_prediction.ipynb` |
| Video / realtime conf. | `0.50` | `video_prediction.ipynb`, `realtime_prediction.ipynb` |
| Classes | `0 body, 1 plane, 2 ship, 3 target` | all prediction notebooks |
| Noise filter | Median → Bilateral → CLAHE (clip 1.2, 16x16) | `noise_filtering.ipynb` |

---

## Environment

- **Author / repo context**: Smart India Hackathon (SIH) project folder.
- **Platform**: Windows (paths in notebooks are hardcoded to `D:\...\OceanScan`).
- **Runtime**: Python 3, CPU-only PyTorch.

> **Note:** The notebooks reference absolute paths under `D:\1. Project Program\1.SIH\OceanScan`. If you move the project, update the `ROOT` variable in each notebook's setup cell.
