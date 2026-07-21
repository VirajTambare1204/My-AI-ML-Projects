"""
Inference Script
================
Run on a single image:
    python predict.py --checkpoint results/<run>/best_model.pth --image path/to/image.jpg

Run on a folder of images:
    python predict.py --checkpoint results/<run>/best_model.pth --folder path/to/images/

Prints predictions and writes results/predictions.csv
"""

import os
import sys
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from model   import CancerCNN, CancerCNNTransfer
from dataset import load_single_image


CLASS_NAMES = ["Benign", "Malignant"]

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ── CLI ───────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Cancer Detection Inference")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--image",  default=None,
                   help="Path to a single image")
    p.add_argument("--folder", default=None,
                   help="Path to a folder of images")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for 'Malignant' prediction (default 0.5)")
    p.add_argument("--device", default="auto")
    p.add_argument("--output_csv", default="results/predictions.csv")
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────
def get_device(choice):
    if choice == "auto":
        if torch.cuda.is_available():          return torch.device("cuda")
        if torch.backends.mps.is_available():  return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


# ── Load model from checkpoint ────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    model_name = args.get("model", "cnn")

    if model_name == "resnet":
        model = CancerCNNTransfer()
    else:
        dropout = args.get("dropout", 0.5)
        model   = CancerCNN(dropout=dropout)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device)


# ── Predict one image ─────────────────────────────────────────────
@torch.no_grad()
def predict_one(model, img_path: str, device: torch.device, threshold: float):
    tensor = load_single_image(img_path).to(device)
    probs  = model.predict_proba(tensor)[0]           # shape (2,)
    mal_prob  = probs[1].item()
    label_idx = 1 if mal_prob >= threshold else 0
    return {
        "image":              img_path,
        "prediction":         CLASS_NAMES[label_idx],
        "benign_prob":        f"{probs[0].item():.4f}",
        "malignant_prob":     f"{mal_prob:.4f}",
        "confidence":         f"{max(probs).item():.4f}",
    }


# ── Main ──────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = get_device(args.device)
    print(f"Device    : {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold : {args.threshold}\n")

    model = load_model(args.checkpoint, device)

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    elif args.folder:
        folder = Path(args.folder)
        image_paths = [
            str(p) for p in sorted(folder.rglob("*"))
            if p.suffix.lower() in SUPPORTED_EXT
        ]
        print(f"Found {len(image_paths)} images in {args.folder}\n")
    else:
        sys.exit("ERROR: Provide --image or --folder")

    # Run predictions
    rows = []
    for path in image_paths:
        try:
            result = predict_one(model, path, device, args.threshold)
            rows.append(result)
            print(f"{'🔴 MALIGNANT' if result['prediction'] == 'Malignant' else '🟢 Benign':20s}"
                  f"  prob={result['malignant_prob']}  "
                  f"conf={result['confidence']}  "
                  f"  {Path(path).name}")
        except Exception as e:
            print(f"  ERROR on {path}: {e}")

    # Save CSV
    if rows:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image", "prediction", "benign_prob",
                      "malignant_prob", "confidence"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nPredictions saved → {out_path}")

        # Summary
        mal = sum(1 for r in rows if r["prediction"] == "Malignant")
        ben = len(rows) - mal
        print(f"\nSummary: {len(rows)} images  |  "
              f"Benign={ben}  Malignant={mal}")


if __name__ == "__main__":
    main()
