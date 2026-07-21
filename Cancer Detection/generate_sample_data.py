"""
generate_sample_data.py
=======================
Creates synthetic histopathology-style images so you can run the
full pipeline without a real dataset.

Run:
    python generate_sample_data.py --n_per_class 50
"""

import argparse
import random
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    raise SystemExit("Pillow is required:  pip install Pillow")


def random_color(base, spread=40):
    return tuple(
        max(0, min(255, b + random.randint(-spread, spread))) for b in base
    )


def draw_cell(draw, cx, cy, r, color, nucleus_color):
    """Draw a simple circular 'cell' with a nucleus."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    nr = r // 3
    draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=nucleus_color)


def make_benign(size=224) -> Image.Image:
    """
    Benign: uniform, regularly spaced cells, soft pinkish background.
    """
    bg_color   = random_color((230, 200, 210))
    cell_color = random_color((200, 150, 170))
    nuc_color  = random_color((90, 50, 110))

    img  = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    spacing = size // 7
    for row in range(7):
        for col in range(7):
            cx = spacing // 2 + col * spacing + random.randint(-4, 4)
            cy = spacing // 2 + row * spacing + random.randint(-4, 4)
            r  = random.randint(12, 16)
            draw_cell(draw, cx, cy, r,
                      random_color(cell_color, 15),
                      random_color(nuc_color,  15))

    return img.filter(ImageFilter.GaussianBlur(0.8))


def make_malignant(size=224) -> Image.Image:
    """
    Malignant: irregular, crowded, variable-size cells, darker & more chaotic.
    """
    bg_color   = random_color((210, 170, 185))
    cell_color = random_color((170, 100, 130))
    nuc_color  = random_color((40,  20,  60))

    img  = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # More cells, irregular placement
    n_cells = random.randint(40, 70)
    for _ in range(n_cells):
        cx = random.randint(10, size - 10)
        cy = random.randint(10, size - 10)
        r  = random.randint(8, 22)   # high size variation → malignant feature
        draw_cell(draw, cx, cy, r,
                  random_color(cell_color, 30),
                  random_color(nuc_color,  20))

    return img.filter(ImageFilter.GaussianBlur(0.5))


# ── CLI ───────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_class", type=int, default=50,
                   help="Images per class per split")
    p.add_argument("--out_dir", default="data",
                   help="Output root (creates train/val/test sub-dirs)")
    p.add_argument("--val_frac",  type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    return p.parse_args()


def main():
    args   = get_args()
    n      = args.n_per_class
    vf, tf = args.val_frac, args.test_frac
    n_val  = max(1, int(n * vf))
    n_test = max(1, int(n * tf))
    n_train = n - n_val - n_test

    splits = {
        "train": n_train,
        "val":   n_val,
        "test":  n_test,
    }
    classes = {
        "benign":    make_benign,
        "malignant": make_malignant,
    }

    total = 0
    for split, count in splits.items():
        for cls, fn in classes.items():
            out_dir = Path(args.out_dir) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = fn()
                img.save(out_dir / f"{cls}_{i:04d}.png")
            total += count
            print(f"  {split}/{cls:12s}  {count} images")

    print(f"\n✓ Generated {total * len(classes)} images total "
          f"under '{args.out_dir}/'")


if __name__ == "__main__":
    main()
