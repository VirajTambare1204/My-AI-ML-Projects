"""
Image Classification System — Entry Point
============================================
Run this file to train, evaluate, or use an image classifier.

Usage:
    python main.py                              → Interactive menu
    python main.py --mode train --arch transfer → Train a transfer-learning model
    python main.py --mode predict --input photo.jpg
    python main.py --mode predict_folder --input test_images/
    python main.py --mode evaluate
    python main.py --mode webcam
"""

import argparse
import sys
import os
from classifier import ImageClassifier


def show_menu():
    """Interactive CLI menu."""
    clf = ImageClassifier()

    while True:
        print("\n" + "=" * 55)
        print("   🖼️   IMAGE CLASSIFICATION SYSTEM")
        print("=" * 55)
        print("  1. Train New Model (Custom CNN)")
        print("  2. Train New Model (Transfer Learning)")
        print("  3. Evaluate Model")
        print("  4. Predict Single Image")
        print("  5. Predict Folder of Images")
        print("  6. Live Webcam Classification")
        print("  7. Exit")
        print("-" * 55)

        choice = input("  Select an option [1-7]: ").strip()

        if choice == "1":
            dataset = input("  Dataset path [dataset]: ").strip() or "dataset"
            epochs = input("  Epochs [15]: ").strip()
            epochs = int(epochs) if epochs.isdigit() else 15

            train_gen, val_gen = clf.load_data(dataset)
            clf.build_custom_cnn(num_classes=len(clf.class_names))
            clf.train(train_gen, val_gen, epochs=epochs)
            clf.evaluate(val_gen)

        elif choice == "2":
            dataset = input("  Dataset path [dataset]: ").strip() or "dataset"
            base = input("  Backbone (mobilenet/resnet50) [mobilenet]: ").strip() or "mobilenet"
            epochs = input("  Epochs [15]: ").strip()
            epochs = int(epochs) if epochs.isdigit() else 15

            train_gen, val_gen = clf.load_data(dataset)
            clf.build_transfer_model(num_classes=len(clf.class_names), base=base)
            clf.train(train_gen, val_gen, epochs=epochs)
            clf.evaluate(val_gen)

        elif choice == "3":
            dataset = input("  Dataset path [dataset]: ").strip() or "dataset"
            clf.load_model()
            _, val_gen = clf.load_data(dataset, augment=False)
            clf.evaluate(val_gen)

        elif choice == "4":
            clf.load_model()
            path = input("  Image path: ").strip()
            results = clf.predict_image(path, top_k=3)
            print("\n  Predictions:")
            for cls, conf in results:
                print(f"    {cls:<20} {conf:.2f}%")

        elif choice == "5":
            clf.load_model()
            folder = input("  Folder path: ").strip()
            results = clf.predict_batch(folder)
            print("\n  Batch Predictions:")
            for fname, preds in results.items():
                cls, conf = preds[0]
                print(f"    {fname:<30} → {cls} ({conf:.1f}%)")

        elif choice == "6":
            clf.load_model()
            run_webcam_classification(clf)

        elif choice == "7":
            print("  Goodbye!")
            sys.exit(0)

        else:
            print("  [!] Invalid option. Please choose 1–7.")


def run_webcam_classification(clf, scale_display=1.0):
    """Real-time classification using webcam feed."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Live classification started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        img = cv2.resize(frame, clf.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = clf.model.predict(img_array, verbose=0)[0]
        top_idx = np.argmax(preds)
        label = clf.class_names[top_idx]
        confidence = preds[top_idx] * 100

        # Draw overlay
        text = f"{label}: {confidence:.1f}%"
        color = (0, 255, 0) if confidence > 60 else (0, 165, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Top-3 bar
        top3_idx = np.argsort(preds)[::-1][:3]
        for i, idx in enumerate(top3_idx):
            bar_text = f"{clf.class_names[idx]}: {preds[idx]*100:.1f}%"
            cv2.putText(frame, bar_text, (10, 75 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Live Image Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Image Classification System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", choices=["menu", "train", "evaluate", "predict", "predict_folder", "webcam"],
                        default="menu", help="Run mode")
    parser.add_argument("--arch", choices=["custom", "transfer"], default="custom",
                        help="Model architecture (for --mode train)")
    parser.add_argument("--backbone", choices=["mobilenet", "resnet50"], default="mobilenet",
                        help="Transfer learning backbone")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--input", type=str, help="Input image or folder path")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (square)")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K predictions to show")

    args = parser.parse_args()

    if args.mode == "menu":
        show_menu()
        return

    clf = ImageClassifier(img_size=(args.img_size, args.img_size))

    if args.mode == "train":
        train_gen, val_gen = clf.load_data(args.dataset, batch_size=args.batch_size)

        if args.arch == "custom":
            clf.build_custom_cnn(num_classes=len(clf.class_names))
        else:
            clf.build_transfer_model(num_classes=len(clf.class_names), base=args.backbone)

        clf.train(train_gen, val_gen, epochs=args.epochs)
        clf.evaluate(val_gen)

    elif args.mode == "evaluate":
        clf.load_model()
        _, val_gen = clf.load_data(args.dataset, augment=False)
        clf.evaluate(val_gen)

    elif args.mode == "predict":
        if not args.input:
            print("[ERROR] --input is required for predict mode.")
            sys.exit(1)
        clf.load_model()
        results = clf.predict_image(args.input, top_k=args.top_k)
        print(f"\nPredictions for {args.input}:")
        for cls, conf in results:
            print(f"  {cls:<20} {conf:.2f}%")

    elif args.mode == "predict_folder":
        if not args.input:
            print("[ERROR] --input (folder) is required for predict_folder mode.")
            sys.exit(1)
        clf.load_model()
        results = clf.predict_batch(args.input)
        print(f"\nBatch predictions for {args.input}:")
        for fname, preds in results.items():
            cls, conf = preds[0]
            print(f"  {fname:<30} → {cls} ({conf:.1f}%)")

    elif args.mode == "webcam":
        clf.load_model()
        run_webcam_classification(clf)


if __name__ == "__main__":
    main()
