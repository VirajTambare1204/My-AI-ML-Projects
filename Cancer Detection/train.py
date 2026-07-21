"""
Training Script
===============
Run:
    python train.py --data_dir data/ --epochs 30 --batch_size 32 --model cnn
    python train.py --data_dir data/ --epochs 20 --model resnet --freeze_backbone

Results are saved under  results/<run_timestamp>/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model   import CancerCNN, CancerCNNTransfer
from dataset import build_loaders
from utils.metrics import compute_metrics, plot_training_curves, save_confusion_matrix


# ── Argument parsing ──────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Train Cancer Detection CNN")
    p.add_argument("--data_dir",       default="data",    help="Root of train/val/test folders")
    p.add_argument("--model",          default="cnn",     choices=["cnn", "resnet"],
                   help="'cnn' = custom CNN   'resnet' = ResNet-18 transfer learning")
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--dropout",        type=float, default=0.5)
    p.add_argument("--freeze_backbone",action="store_true",
                   help="Freeze ResNet backbone (only for --model resnet)")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--patience",       type=int, default=7,
                   help="Early-stopping patience (epochs with no val improvement)")
    p.add_argument("--save_dir",       default="results",  help="Output directory")
    p.add_argument("--device",         default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'")
    return p.parse_args()


# ── Device selection ──────────────────────────────────────────────
def get_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():    return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


# ── One epoch ─────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

            all_preds .extend(preds .cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs .extend(probs .cpu().tolist())

    avg_loss = total_loss / total
    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    metrics["accuracy"] = correct / total
    return metrics


# ── Training loop ─────────────────────────────────────────────────
def train(args):
    # ── Setup ──────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Run ID   : {run_id}")
    print(f"  Save dir : {save_dir}")
    print(f"{'='*60}\n")

    device = get_device(args.device)
    print(f"Device: {device}\n")

    # ── Data ───────────────────────────────────────────────────────
    loaders, _ = build_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    assert "train" in loaders, "No 'train' split found under data_dir!"
    assert "val"   in loaders, "No 'val'   split found under data_dir!"

    # ── Model ──────────────────────────────────────────────────────
    if args.model == "cnn":
        model = CancerCNN(dropout=args.dropout)
    else:
        model = CancerCNNTransfer(freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Optimizer / scheduler / loss ───────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3,
                                   factor=0.5)

    # ── History & early-stop bookkeeping ──────────────────────────
    history   = {"train": [], "val": []}
    best_val_auc  = 0.0
    best_epoch    = 0
    no_improve    = 0

    # ── Epoch loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, loaders["train"], criterion,
                            optimizer, device, train=True)
        val_m   = run_epoch(model, loaders["val"],   criterion,
                            optimizer, device, train=False)

        scheduler.step(val_m["auc"])

        history["train"].append(train_m)
        history["val"]  .append(val_m)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"[{elapsed:.1f}s]  "
              f"train_loss={train_m['loss']:.4f}  train_acc={train_m['accuracy']:.4f}  "
              f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}  "
              f"val_auc={val_m['auc']:.4f}")

        # ── Checkpoint ────────────────────────────────────────────
        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            best_epoch   = epoch
            no_improve   = 0
            ckpt = {
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_auc":     best_val_auc,
                "args":        vars(args),
            }
            torch.save(ckpt, save_dir / "best_model.pth")
            print(f"  ✓ Saved best model  (AUC={best_val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    print(f"\n✓ Training complete — best epoch={best_epoch}, "
          f"best val AUC={best_val_auc:.4f}")

    # ── Save history ───────────────────────────────────────────────
    import numpy as np
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items() if k != "conf_matrix"}
        if isinstance(obj, list):
            return [_json_safe(i) for i in obj]
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj
    with open(save_dir / "history.json", "w") as f:
        json.dump(_json_safe(history), f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────
    plot_training_curves(history, save_dir / "training_curves.png")

    # ── Test evaluation ───────────────────────────────────────────
    if "test" in loaders:
        print("\nEvaluating on test set …")
        # reload best checkpoint
        ckpt = torch.load(save_dir / "best_model.pth", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_m = run_epoch(model, loaders["test"], criterion,
                           optimizer, device, train=False)
        print(f"Test  accuracy={test_m['accuracy']:.4f}  "
              f"AUC={test_m['auc']:.4f}  "
              f"F1={test_m['f1']:.4f}  "
              f"sensitivity={test_m['sensitivity']:.4f}  "
              f"specificity={test_m['specificity']:.4f}")
        save_confusion_matrix(test_m["conf_matrix"],
                              ["Benign", "Malignant"],
                              save_dir / "confusion_matrix.png")
        with open(save_dir / "test_results.json", "w") as f:
            # conf_matrix isn't JSON serialisable directly
            test_m_json = {k: v for k, v in test_m.items()
                           if k != "conf_matrix"}
            json.dump(test_m_json, f, indent=2)

    print(f"\nAll outputs saved to: {save_dir}")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    train(get_args())
