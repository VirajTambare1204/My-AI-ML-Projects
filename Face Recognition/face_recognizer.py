"""
Face Recognition System
=======================
Real-time face detection and recognition using OpenCV and face_recognition library.
Supports: webcam feed, image recognition, and dataset training.
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
from pathlib import Path


class FaceRecognitionSystem:
    def __init__(self, encodings_path="encodings/face_encodings.pkl"):
        self.encodings_path = encodings_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

    # ─────────────────────────────────────────────
    # Encoding Management
    # ─────────────────────────────────────────────

    def load_encodings(self):
        """Load saved face encodings from disk."""
        if os.path.exists(self.encodings_path):
            print(f"[INFO] Loading encodings from {self.encodings_path}")
            with open(self.encodings_path, "rb") as f:
                data = pickle.load(f)
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]
            print(f"[INFO] Loaded {len(self.known_face_names)} face(s): {set(self.known_face_names)}")
        else:
            print("[INFO] No saved encodings found. Train the model first.")

    def save_encodings(self):
        """Persist face encodings to disk."""
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
        with open(self.encodings_path, "wb") as f:
            pickle.dump({"encodings": self.known_face_encodings, "names": self.known_face_names}, f)
        print(f"[INFO] Encodings saved to {self.encodings_path}")

    def train_from_directory(self, dataset_path="dataset"):
        """
        Train the model from a folder of images.
        Expected structure:
            dataset/
                PersonName/
                    img1.jpg
                    img2.jpg
        """
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset directory '{dataset_path}' not found.")
            return

        self.known_face_encodings = []
        self.known_face_names = []

        people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if not people:
            print("[ERROR] No subdirectories found in dataset folder.")
            return

        print(f"[INFO] Training on {len(people)} person(s): {people}")

        for person_name in people:
            person_dir = os.path.join(dataset_path, person_name)
            image_files = [
                f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]

            print(f"  ↳ {person_name}: {len(image_files)} image(s)")

            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(person_name)
                else:
                    print(f"    [WARN] No face found in {img_file}, skipping.")

        self.save_encodings()
        print(f"[INFO] Training complete. Total encodings: {len(self.known_face_encodings)}")

    def add_face_from_webcam(self, person_name, save_count=5, dataset_path="dataset"):
        """Capture face samples from webcam and add to dataset."""
        save_dir = os.path.join(dataset_path, person_name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        print(f"[INFO] Capturing {save_count} face samples for '{person_name}'. Press 'c' to capture, 'q' to quit.")
        captured = 0

        while captured < save_count:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")

            for (top, right, bottom, left) in boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            overlay = frame.copy()
            cv2.putText(overlay, f"Capturing for: {person_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, f"Saved: {captured}/{save_count} | Press 'c' to capture",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Add Face - Press 'c' to capture", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and boxes:
                img_path = os.path.join(save_dir, f"{person_name}_{captured + 1}.jpg")
                cv2.imwrite(img_path, frame)
                captured += 1
                print(f"  ↳ Saved sample {captured}/{save_count}")
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured > 0:
            print(f"[INFO] Captured {captured} images. Running training...")
            self.train_from_directory(dataset_path)

    # ─────────────────────────────────────────────
    # Recognition
    # ─────────────────────────────────────────────

    def recognize_faces_in_image(self, image_path, output_path=None):
        """Recognize faces in a single image file."""
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return

        image = face_recognition.load_image_file(image_path)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(image, model="hog")
        face_encodings = face_recognition.face_encodings(image, face_locations)

        print(f"[INFO] Found {len(face_locations)} face(s) in {image_path}")

        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            name, confidence = self._identify_face(enc)
            self._draw_label(bgr, top, right, bottom, left, name, confidence)
            print(f"  ↳ {name} ({confidence:.1f}% confidence)")

        cv2.imshow("Face Recognition - Image", bgr)

        if output_path:
            cv2.imwrite(output_path, bgr)
            print(f"[INFO] Result saved to {output_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def recognize_realtime(self, scale=0.5, tolerance=0.5):
        """
        Real-time face recognition from webcam.
        - scale: resize factor for faster processing (0.25–1.0)
        - tolerance: lower = stricter matching (0.4–0.6 recommended)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        print("[INFO] Real-time recognition started. Press 'q' to quit.")
        process_this_frame = True
        face_locations, face_names, confidences = [], [], []
        fps_start = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if process_this_frame:
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small, model="hog")
                face_encodings_list = face_recognition.face_encodings(rgb_small, face_locations)

                face_names = []
                confidences = []
                for enc in face_encodings_list:
                    name, conf = self._identify_face(enc, tolerance)
                    face_names.append(name)
                    confidences.append(conf)

            process_this_frame = not process_this_frame

            # Draw results
            for (top, right, bottom, left), name, conf in zip(face_locations, face_names, confidences):
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                self._draw_label(frame, top, right, bottom, left, name, conf)

            # FPS counter
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(face_names)} | Press Q to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Face Recognition - Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam closed.")

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _identify_face(self, encoding, tolerance=0.5):
        """Match encoding against known faces. Returns (name, confidence%)."""
        if not self.known_face_encodings:
            return "Unknown", 0.0

        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]

        if best_dist <= tolerance:
            name = self.known_face_names[best_idx]
            confidence = (1 - best_dist) * 100
        else:
            name = "Unknown"
            confidence = 0.0

        return name, confidence

    def _draw_label(self, frame, top, right, bottom, left, name, confidence):
        """Draw bounding box and name label on frame."""
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = f"{name}" if name == "Unknown" else f"{name} ({confidence:.0f}%)"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (left, bottom - 25), (left + label_size[0] + 6, bottom), color, -1)
        cv2.putText(frame, label, (left + 3, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    def list_known_faces(self):
        """Print all known persons."""
        names = sorted(set(self.known_face_names))
        if names:
            print(f"[INFO] Known faces ({len(names)}): {', '.join(names)}")
        else:
            print("[INFO] No known faces trained yet.")
