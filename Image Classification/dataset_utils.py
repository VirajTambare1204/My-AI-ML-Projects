"""
Dataset Preparation Utility
============================
Tools to prepare, verify, and clean datasets for image classification.
"""

import os
import shutil
import random


def verify_dataset(dataset_path="dataset"):
    """Check dataset structure and report image counts per class."""
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path '{dataset_path}' does not exist.")
        return

    print(f"\n{'='*55}")
    print(f"  Dataset Report: {dataset_path}")
    print(f"{'='*55}")

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    if not classes:
        print("  No class folders found.\n")
        return

    total = 0
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_ext)]
        total += len(images)
        status = "OK " if len(images) >= 50 else ("LOW" if len(images) >= 10 else "BAD")
        print(f"  [{status}] {cls:<20} {len(images)} images")

    print(f"{'-'*55}")
    print(f"  Total: {len(classes)} classes | {total} images")
    avg = total / len(classes) if classes else 0
    print(f"  Average per class: {avg:.0f}")

    if avg < 50:
        print("\n  Recommendation: aim for 50+ images per class for good accuracy.")
        print("  Consider using transfer learning (--arch transfer) for small datasets.")
    print(f"{'='*55}\n")


def split_train_test(source_dir, output_dir="dataset_split", test_ratio=0.15, seed=42):
    """
    Split a flat dataset (class folders with all images) into train/test directories.

    Output structure:
        dataset_split/
            train/
                class1/...
            test/
                class1/...
    """
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for cls in classes:
        src_cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(valid_ext)]
        random.shuffle(images)

        split_idx = int(len(images) * (1 - test_ratio))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(src_cls_dir, img), os.path.join(train_dir, cls, img))
        for img in test_images:
            shutil.copy2(os.path.join(src_cls_dir, img), os.path.join(test_dir, cls, img))

        print(f"  -> {cls}: {len(train_images)} train / {len(test_images)} test")

    print(f"\n[INFO] Split complete. Output: {output_dir}/")


def create_sample_structure(dataset_path="dataset", classes=None):
    """Create a sample dataset folder structure with instructions."""
    if classes is None:
        classes = ["class_1", "class_2", "class_3"]

    for cls in classes:
        os.makedirs(os.path.join(dataset_path, cls), exist_ok=True)

    readme = f"""HOW TO BUILD YOUR DATASET
==========================
1. Each subfolder name = a class label.
   Example:
     dataset/
       cats/
         img1.jpg
         img2.jpg
       dogs/
         img1.jpg
         img2.jpg

2. Recommended: 50-200+ images per class for good accuracy.
   - At least 20 images minimum if using transfer learning.

3. Images can be .jpg, .jpeg, .png, .bmp, or .webp

4. Verify your dataset:
   python dataset_utils.py verify

5. Train the model:
   python main.py --mode train --arch transfer --epochs 15

Current sample classes created: {classes}
Replace these folders with your real class names and images.
"""
    with open(os.path.join(dataset_path, "README.txt"), "w") as f:
        f.write(readme)

    print(f"[INFO] Sample dataset structure created at ./{dataset_path}/")
    print(f"       Classes: {classes}")
    print(f"       Add images to each folder, then run: python main.py --mode train")


def remove_corrupted_images(dataset_path="dataset"):
    """Scan dataset and remove unreadable/corrupted image files."""
    from PIL import Image

    removed = 0
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith(valid_ext):
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except Exception:
                    print(f"  -> Removing corrupted file: {fpath}")
                    os.remove(fpath)
                    removed += 1

    print(f"\n[INFO] Scan complete. Removed {removed} corrupted file(s).")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "verify":
            verify_dataset()
        elif cmd == "create":
            create_sample_structure()
        elif cmd == "clean":
            remove_corrupted_images()
        elif cmd == "split":
            src = sys.argv[2] if len(sys.argv) > 2 else "dataset"
            split_train_test(src)
        else:
            print("Usage: python dataset_utils.py [verify|create|clean|split <source_dir>]")
    else:
        print("\nDataset Utility Commands:")
        print("  python dataset_utils.py verify          - Check dataset health")
        print("  python dataset_utils.py create           - Create sample folder structure")
        print("  python dataset_utils.py clean             - Remove corrupted images")
        print("  python dataset_utils.py split <dir>      - Split into train/test sets")
        verify_dataset()
