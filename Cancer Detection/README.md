# Cancer Detection with CNN 🔬

A complete, executable deep-learning project that classifies histopathology
images as **Benign** or **Malignant** using a Convolutional Neural Network.

---

## Project Structure

```
cancer_detection/
├── model.py                  ← CNN architecture (custom + ResNet-18 transfer)
├── dataset.py                ← DataLoader, augmentation, class balancing
├── train.py                  ← Training loop with early stopping & checkpointing
├── evaluate.py               ← Full evaluation: AUC, F1, sensitivity, specificity
├── predict.py                ← Inference on single image or folder
├── generate_sample_data.py   ← Synthetic data generator (for demo/testing)
├── requirements.txt
├── utils/
│   ├── __init__.py
│   └── metrics.py            ← AUC, F1, confusion matrix, ROC-curve plotting
└── data/                     ← YOUR dataset goes here
    ├── train/
    │   ├── benign/
    │   └── malignant/
    ├── val/
    │   ├── benign/
    │   └── malignant/
    └── test/
        ├── benign/
        └── malignant/
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data  *(or generate synthetic demo data)*

**Option A — Use a real dataset** (e.g. BreaKHis, PCam, IDC, TCGA):
Place images in `data/train/benign/`, `data/train/malignant/`, etc.

**Option B — Generate synthetic images** (for testing the pipeline):
```bash
python generate_sample_data.py --n_per_class 100
```

### 3. Train

```bash
# Custom CNN (default)
python train.py --data_dir data/ --epochs 30 --batch_size 32

# ResNet-18 transfer learning (recommended for small datasets)
python train.py --data_dir data/ --model resnet --epochs 20 --batch_size 32

# With frozen backbone (only trains the classifier head)
python train.py --data_dir data/ --model resnet --freeze_backbone --epochs 10
```

### 4. Evaluate on the test set

```bash
python evaluate.py \
    --checkpoint results/<run_id>/best_model.pth \
    --data_dir   data/ \
    --split      test
```

### 5. Predict

```bash
# Single image
python predict.py \
    --checkpoint results/<run_id>/best_model.pth \
    --image      path/to/slide.jpg

# Whole folder
python predict.py \
    --checkpoint results/<run_id>/best_model.pth \
    --folder     path/to/images/ \
    --output_csv results/predictions.csv
```

---

## Model Architecture

### Custom CNN (`--model cnn`)

```
Input (3 × 224 × 224)
    ↓
[Conv 3×3 → BN → ReLU] × 2  →  MaxPool    # 32 filters  → 112×112
    ↓
[Conv 3×3 → BN → ReLU] × 2  →  MaxPool    # 64 filters  →  56×56
    ↓
[Conv 3×3 → BN → ReLU] × 2  →  MaxPool    # 128 filters →  28×28
    ↓
[Conv 3×3 → BN → ReLU] × 2  →  MaxPool    # 256 filters →  14×14
    ↓
Global Average Pooling  →  256-d vector
    ↓
Dropout → FC(256→128) → ReLU → Dropout → FC(128→2)
    ↓
Output logits  [Benign, Malignant]
```

### Transfer Learning (`--model resnet`)

- **Backbone**: ResNet-18 pre-trained on ImageNet
- **Head replaced**: Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→2)
- Optionally freeze the backbone with `--freeze_backbone`

---

## Training Details

| Setting            | Default                   |
|--------------------|---------------------------|
| Optimizer          | Adam                      |
| Learning rate      | 1e-3                      |
| LR scheduler       | ReduceLROnPlateau (×0.5)  |
| Weight decay       | 1e-4                      |
| Early stopping     | 7 epochs patience         |
| Class balancing    | WeightedRandomSampler     |
| Augmentations      | Flip, rotate, color jitter|
| Input size         | 224 × 224                 |

---

## Metrics Reported

| Metric      | Why it matters for cancer detection                  |
|-------------|------------------------------------------------------|
| Accuracy    | Overall correctness                                  |
| AUC-ROC     | Discrimination ability regardless of threshold       |
| F1 Score    | Harmonic mean of precision and recall                |
| Sensitivity | True-positive rate (catch as many cancers as possible)|
| Specificity | True-negative rate (avoid unnecessary biopsies)       |

---

## Recommended Public Datasets

| Dataset | Description                              | Link |
|---------|------------------------------------------|------|
| BreaKHis | Breast tumour histopathology (40×–400×)| [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) |
| PCam     | Patch Camelyon (lymph node sections)   | [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection) |
| IDC      | Invasive Ductal Carcinoma patches       | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) |

---

## Output Files (in `results/<run_id>/`)

```
best_model.pth          ← Best checkpoint (highest val AUC)
history.json            ← Per-epoch metrics
training_curves.png     ← Loss, accuracy, AUC, F1 plots
confusion_matrix.png    ← Test confusion matrix
roc_curve.png           ← ROC curve with AUC
test_results.json       ← Final test metrics
predictions.csv         ← Per-image predictions (from predict.py)
```

---

## Disclaimer

This project is for **educational and research purposes only**.  
It is **not** a medical device and must not be used for clinical diagnosis.
