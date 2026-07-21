"""
Dataset Preparation Utility
============================
Helps you build a clean dataset for training the face recognition model.
- Download sample images from URLs
- Verify and preview dataset structure
- Augment existing images for better accuracy
"""

import os
import cv2
import numpy as np
import face_recognition
from pathlib import Path


def verify_dataset(dataset_path="dataset"):
    """Check dataset structure and report face counts per person."""
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path '{dataset_path}' does not exist.")
        return

    print(f"\n{'='*50}")
    print(f"  Dataset Report: {dataset_path}")
    print(f"{'='*50}")

    total_images = 0
    total_faces = 0
    people = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    if not people:
        print("  No person folders found.\n")
        return

    for person in people:
        person_dir = os.path.join(dataset_path, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        face_count = 0

        for img_file in images:
            img = face_recognition.load_image_file(os.path.join(person_dir, img_file))
            encs = face_recognition.face_encodings(img)
            if encs:
                face_count += 1

        total_images += len(images)
        total_faces += face_count
        status = "✅" if face_count >= 3 else "⚠️ (add more images)"
        print(f"  {status}  {person}: {len(images)} images | {face_count} usable faces")

    print(f"{'─'*50}")
    print(f"  Total: {len(people)} people | {total_images} images | {total_faces} usable faces")
    print(f"{'='*50}\n")


def augment_images(dataset_path="dataset", augment_count=3):
    """
    Augment dataset images with basic flips and brightness variations
    to improve recognition accuracy with small datasets.
    """
    print(f"[INFO] Augmenting dataset at '{dataset_path}' (×{augment_count} per image)...")

    people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for person in people:
        person_dir = os.path.join(dataset_path, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
                  and "_aug" not in f]

        for img_file in images:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            base = Path(img_file).stem
            augmented = []

            # Horizontal flip
            augmented.append(("flip", cv2.flip(img, 1)))

            # Brightness variations
            for i, alpha in enumerate([0.75, 1.3]):
                bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                augmented.append((f"bright{i}", bright))

            for suffix, aug_img in augmented[:augment_count]:
                out_path = os.path.join(person_dir, f"{base}_aug_{suffix}.jpg")
                if not os.path.exists(out_path):
                    cv2.imwrite(out_path, aug_img)

        print(f"  ↳ {person}: augmented {len(images)} source image(s)")

    print("[INFO] Augmentation complete.")


def create_sample_dataset():
    """
    Create a sample dataset folder structure with placeholder instructions.
    Use this as a starting point.
    """
    os.makedirs("dataset/Person1", exist_ok=True)
    os.makedirs("dataset/Person2", exist_ok=True)

    readme = """HOW TO ADD YOUR OWN FACES
=========================
1. Create a subfolder with the person's name inside 'dataset/'
   Example: dataset/John_Doe/

2. Add 5–20 clear face images (JPG/PNG) of that person.
   - Good lighting, different angles
   - Only ONE face visible per image ideally

3. Run training:
   python main.py --mode train

4. Then run recognition:
   python main.py --mode realtime
"""
    with open("dataset/README.txt", "w") as f:
        f.write(readme)

    print("[INFO] Sample dataset structure created at ./dataset/")
    print("       Add face images to dataset/Person1/ and dataset/Person2/")
    print("       then run: python main.py --mode train")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "verify":
            verify_dataset()
        elif cmd == "augment":
            augment_images()
        elif cmd == "create":
            create_sample_dataset()
        else:
            print("Usage: python dataset_utils.py [verify|augment|create]")
    else:
        print("\nDataset Utility Commands:")
        print("  python dataset_utils.py verify   — Check dataset health")
        print("  python dataset_utils.py augment  — Augment images for better accuracy")
        print("  python dataset_utils.py create   — Create sample folder structure")
        verify_dataset()
