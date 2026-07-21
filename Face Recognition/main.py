"""
Face Recognition System — Entry Point
======================================
Run this file to launch the face recognition system.

Usage:
    python main.py                    → Interactive menu
    python main.py --mode realtime    → Live webcam recognition
    python main.py --mode image --input photo.jpg
    python main.py --mode train --dataset dataset/
    python main.py --mode add --name "John"
"""

import argparse
import sys
from face_recognizer import FaceRecognitionSystem


def show_menu(frs: FaceRecognitionSystem):
    """Interactive CLI menu."""
    while True:
        print("\n" + "=" * 50)
        print("   🎭  FACE RECOGNITION SYSTEM")
        print("=" * 50)
        print("  1. Real-time Recognition (Webcam)")
        print("  2. Recognize Faces in Image")
        print("  3. Train Model from Dataset Folder")
        print("  4. Add New Person via Webcam")
        print("  5. List Known Faces")
        print("  6. Exit")
        print("-" * 50)

        choice = input("  Select an option [1-6]: ").strip()

        if choice == "1":
            frs.recognize_realtime()

        elif choice == "2":
            path = input("  Enter image path: ").strip()
            save = input("  Save output? (y/n): ").strip().lower()
            out = "output/result.jpg" if save == "y" else None
            if out:
                import os; os.makedirs("output", exist_ok=True)
            frs.recognize_faces_in_image(path, output_path=out)

        elif choice == "3":
            path = input("  Dataset directory path [dataset]: ").strip() or "dataset"
            frs.train_from_directory(path)

        elif choice == "4":
            name = input("  Enter person's name: ").strip()
            count = input("  Number of samples to capture [5]: ").strip()
            count = int(count) if count.isdigit() else 5
            frs.add_face_from_webcam(name, save_count=count)

        elif choice == "5":
            frs.list_known_faces()

        elif choice == "6":
            print("  Goodbye!")
            sys.exit(0)

        else:
            print("  [!] Invalid option. Please choose 1–6.")


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", choices=["menu", "realtime", "image", "train", "add", "list"],
                        default="menu", help="Run mode")
    parser.add_argument("--input", type=str, help="Input image path (for --mode image)")
    parser.add_argument("--output", type=str, help="Output image path (for --mode image)")
    parser.add_argument("--name", type=str, help="Person name (for --mode add)")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset path (for --mode train)")
    parser.add_argument("--samples", type=int, default=5, help="Number of webcam samples (for --mode add)")
    parser.add_argument("--scale", type=float, default=0.5, help="Frame scale for realtime (0.25–1.0)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Match tolerance (0.4–0.6)")
    parser.add_argument("--encodings", type=str, default="encodings/face_encodings.pkl",
                        help="Path to encodings file")

    args = parser.parse_args()
    frs = FaceRecognitionSystem(encodings_path=args.encodings)

    if args.mode == "menu":
        show_menu(frs)

    elif args.mode == "realtime":
        frs.recognize_realtime(scale=args.scale, tolerance=args.tolerance)

    elif args.mode == "image":
        if not args.input:
            print("[ERROR] --input is required for image mode.")
            sys.exit(1)
        frs.recognize_faces_in_image(args.input, output_path=args.output)

    elif args.mode == "train":
        frs.train_from_directory(args.dataset)

    elif args.mode == "add":
        if not args.name:
            args.name = input("Enter person's name: ").strip()
        frs.add_face_from_webcam(args.name, save_count=args.samples)

    elif args.mode == "list":
        frs.list_known_faces()


if __name__ == "__main__":
    main()
