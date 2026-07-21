# 🎭 Face Recognition AI/ML System

A complete, executable face recognition system built with Python, OpenCV, and `face_recognition` (dlib-based). Supports real-time webcam recognition, image-based recognition, model training from a photo dataset, and automated attendance logging.

---

## 📁 Project Structure

```
face_recognition_project/
├── main.py              ← Entry point (CLI + interactive menu)
├── face_recognizer.py   ← Core recognition engine
├── attendance.py        ← Attendance logger with CSV export
├── dataset_utils.py     ← Dataset tools (verify, augment, scaffold)
├── requirements.txt     ← Python dependencies
├── dataset/             ← Your training images go here
│   ├── Person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── Person2/
│       └── img1.jpg
├── encodings/           ← Auto-generated encoding cache
│   └── face_encodings.pkl
└── attendance_logs/     ← Auto-generated attendance CSVs
    └── attendance_2025-01-01.csv
```

---

## ⚙️ Installation

### 1. Prerequisites

- Python 3.8–3.11
- CMake (required for dlib)
  - **Windows**: `winget install Kitware.CMake`
  - **macOS**: `brew install cmake`
  - **Linux**: `sudo apt install cmake build-essential`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` compilation takes 5–10 minutes on first install. This is normal.

### 3. Verify Installation

```bash
python -c "import face_recognition; import cv2; print('✅ All packages OK')"
```

---

## 🚀 Quick Start

### Option A: Interactive Menu

```bash
python main.py
```

Follow the on-screen menu to train, recognize, or add faces.

### Option B: Command Line

```bash
# Real-time webcam recognition
python main.py --mode realtime

# Recognize faces in an image
python main.py --mode image --input photo.jpg --output result.jpg

# Train model from dataset folder
python main.py --mode train --dataset dataset/

# Add a new person via webcam
python main.py --mode add --name "Vaishnavi" --samples 8

# List all known faces
python main.py --mode list
```

### Option C: Attendance System

```bash
python attendance.py
```

Automatically marks attendance for recognized faces and logs to `attendance_logs/attendance_YYYY-MM-DD.csv`.

---

## 🏋️ Training Your Model

1. **Create folder structure:**
   ```
   dataset/
     Vaishnavi/   ← folder name becomes the recognized label
       img1.jpg
       img2.jpg
       img3.jpg
     John/
       photo1.jpg
   ```

2. **Run training:**
   ```bash
   python main.py --mode train
   ```

3. **Verify dataset health:**
   ```bash
   python dataset_utils.py verify
   ```

4. **Augment images** (if you have fewer than 5 per person):
   ```bash
   python dataset_utils.py augment
   ```

---

## 🎛️ CLI Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--mode` | `menu` | `menu`, `realtime`, `image`, `train`, `add`, `list` |
| `--input` | — | Image path (for `image` mode) |
| `--output` | — | Save result image path |
| `--name` | — | Person name (for `add` mode) |
| `--dataset` | `dataset/` | Training dataset path |
| `--samples` | `5` | Webcam capture count (for `add` mode) |
| `--scale` | `0.5` | Frame resize scale for speed (0.25–1.0) |
| `--tolerance` | `0.5` | Match strictness (lower = stricter, 0.4–0.6) |
| `--encodings` | `encodings/face_encodings.pkl` | Encoding file path |

---

## 🔧 Tips for Best Accuracy

- **Dataset size:** 5–20 images per person is ideal
- **Image quality:** Good lighting, front-facing, varied angles
- **Tolerance:** Use `0.45` for stricter matching, `0.6` if too many unknowns
- **Scale:** `0.25` for faster processing on slow machines; `1.0` for max accuracy
- **Augmentation:** Run `python dataset_utils.py augment` if dataset is small

---

## 📦 Key Technologies

| Library | Purpose |
|---|---|
| `face_recognition` | Face detection & encoding (dlib HOG + CNN) |
| `opencv-python` | Webcam capture, drawing, image I/O |
| `numpy` | Distance calculations, array ops |
| `pickle` | Encoding persistence |

---

## 🗂️ Output Files

- `encodings/face_encodings.pkl` — Trained face encodings cache
- `attendance_logs/attendance_YYYY-MM-DD.csv` — Daily attendance records
- `output/result.jpg` — Annotated output images (when `--output` is set)
