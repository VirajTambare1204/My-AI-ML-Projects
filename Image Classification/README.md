# 🖼️ Image Classification AI/ML System

A complete, executable image classification system built with TensorFlow/Keras. Supports training from scratch with a custom CNN, transfer learning (MobileNetV2/ResNet50), evaluation with confusion matrices, batch prediction, and real-time webcam classification.

---

## 📁 Project Structure

```
image_classification_project/
├── main.py              ← Entry point (CLI + interactive menu)
├── classifier.py        ← Core model engine (build, train, evaluate, predict)
├── dataset_utils.py     ← Dataset tools (verify, split, clean, scaffold)
├── requirements.txt     ← Python dependencies
├── dataset/             ← Your training images go here
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       └── img1.jpg
└── saved_model/          ← Auto-generated after training
    ├── final_model.keras
    ├── class_names.json
    ├── training_history.png
    └── confusion_matrix.png
```

---

## ⚙️ Installation

### 1. Prerequisites
- Python 3.9–3.11
- (Optional) NVIDIA GPU + CUDA for faster training

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install tensorflow[and-cuda]
```

### 3. Verify Installation

```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

---

## 🚀 Quick Start

### Option A: Interactive Menu

```bash
python main.py
```

### Option B: Command Line

```bash
# Train with a custom CNN (built from scratch)
python main.py --mode train --arch custom --epochs 20

# Train with transfer learning (recommended — faster, more accurate)
python main.py --mode train --arch transfer --backbone mobilenet --epochs 15

# Evaluate the trained model
python main.py --mode evaluate

# Predict a single image
python main.py --mode predict --input photo.jpg --top_k 3

# Predict all images in a folder
python main.py --mode predict_folder --input test_images/

# Real-time webcam classification
python main.py --mode webcam
```

---

## 🏋️ Building Your Dataset

1. **Create folder structure:**
   ```
   dataset/
     cats/
       img1.jpg
       img2.jpg
     dogs/
       img1.jpg
       img2.jpg
   ```
   Folder names become the class labels automatically.

2. **Scaffold a starter structure:**
   ```bash
   python dataset_utils.py create
   ```

3. **Verify dataset health:**
   ```bash
   python dataset_utils.py verify
   ```

4. **Remove corrupted images:**
   ```bash
   python dataset_utils.py clean
   ```

5. **Train:**
   ```bash
   python main.py --mode train --arch transfer
   ```

---

## 🎛️ CLI Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `--mode` | `menu` | `menu`, `train`, `evaluate`, `predict`, `predict_folder`, `webcam` |
| `--arch` | `custom` | `custom` (CNN from scratch) or `transfer` (pretrained backbone) |
| `--backbone` | `mobilenet` | `mobilenet` or `resnet50` (for transfer learning) |
| `--dataset` | `dataset/` | Path to training dataset |
| `--input` | — | Image or folder path (for predict modes) |
| `--epochs` | `15` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--img_size` | `224` | Input image size (square) |
| `--top_k` | `3` | Number of top predictions to display |

---

## 🧠 Choosing an Architecture

| Architecture | Best For | Speed | Accuracy (small data) |
|---|---|---|---|
| **Custom CNN** | Learning, large datasets (1000+/class) | Medium | Lower on small data |
| **Transfer: MobileNetV2** | Most projects, fast inference | Fast | High |
| **Transfer: ResNet50** | Maximum accuracy, complex images | Slower | Highest |

**Recommendation:** Use `--arch transfer --backbone mobilenet` unless you have a large dataset (1000+ images per class) or need maximum control.

---

## 📊 Outputs After Training

- `saved_model/final_model.keras` — Trained model weights
- `saved_model/best_model.keras` — Best checkpoint during training
- `saved_model/class_names.json` — Class label mapping
- `saved_model/training_history.png` — Accuracy/loss curves
- `saved_model/confusion_matrix.png` — Per-class performance breakdown

---

## 🔧 Tips for Best Results

- **Dataset size:** 50–200+ images per class minimum; more is better
- **Class balance:** Keep similar image counts across classes
- **Image quality:** Clear, well-lit, varied backgrounds and angles
- **Small dataset?** Always use `--arch transfer` — pretrained weights compensate for limited data
- **Overfitting?** Increase dropout, add more augmentation, or reduce epochs
- **Underfitting?** Train longer, unfreeze backbone layers (fine-tuning), or use a bigger backbone

---

## 📦 Key Technologies

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | Model building, training, inference |
| `opencv-python` | Webcam capture for live classification |
| `scikit-learn` | Classification report, confusion matrix |
| `matplotlib` / `seaborn` | Training curves and visualization |
| `pillow` | Image loading and corruption checks |
