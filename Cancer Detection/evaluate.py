"""
Evaluation Script
=================
Full evaluation on any split (test by default).

Run:
    python evaluate.py --checkpoint results/<run>/best_model.pth --data_dir data/ --split test
"""

import argparse
import json
from pathlib import Path

import torch

from model   import CancerCNN, CancerCNNTransfer
from dataset import build_loaders
from utils.metrics import (
    compute_metrics,
    save_confusion_matrix,
    plot_roc_curve,
)


def get_args():
    p = argparse.ArgumentParser(description="Evaluate Cancer Detection CNN")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--device",     default="auto")
    p.add_argument("--output_dir", default=None,
                   help="Where to save plots/report (default: same dir as checkpoint)")
    return p.parse_args()


def get_device(choice):
    if choice == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def main():
    args   = get_args()
    device = get_device(args.device)
    print(f"Device : {device}")

    # ── Load model ─────────────────────────────────────────────────
    ckpt   = torch.load(args.checkpoint, map_location=device)
    a      = ckpt.get("args", {})
    if a.get("model", "cnn") == "resnet":
        model = CancerCNNTransfer()
    else:
        model = CancerCNN(dropout=a.get("dropout", 0.5))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"Loaded checkpoint (epoch={ckpt.get('epoch','?')}, "
          f"val_auc={ckpt.get('val_auc', 0):.4f})")

    # ── Load data ──────────────────────────────────────────────────
    loaders, _ = build_loaders(args.data_dir, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    if args.split not in loaders:
        raise ValueError(f"Split '{args.split}' not found in {args.data_dir}")
    loader = loaders[args.split]

    # ── Collect predictions ────────────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = model.predict_proba(imgs)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_preds .extend(preds .cpu().tolist())
            all_probs .extend(probs[:, 1].cpu().tolist())

    # ── Compute metrics ────────────────────────────────────────────
    m = compute_metrics(all_labels, all_preds, all_probs)

    correct  = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)

    print(f"\n{'='*50}")
    print(f"  Split      : {args.split}  ({len(all_labels)} samples)")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  AUC-ROC    : {m['auc']:.4f}")
    print(f"  F1 Score   : {m['f1']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Sensitivity: {m['sensitivity']:.4f}")
    print(f"  Specificity: {m['specificity']:.4f}")
    print(f"{'='*50}")

    # ── Save outputs ───────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir \
              else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        m["conf_matrix"],
        ["Benign", "Malignant"],
        out_dir / f"{args.split}_confusion_matrix.png",
    )
    plot_roc_curve(all_labels, all_probs,
                   out_dir / f"{args.split}_roc_curve.png")

    report = {
        "split":       args.split,
        "n_samples":   len(all_labels),
        "accuracy":    accuracy,
        "auc":         m["auc"],
        "f1":          m["f1"],
        "precision":   m["precision"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
    }
    report_path = out_dir / f"{args.split}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
