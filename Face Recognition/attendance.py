"""
Attendance Logger
==================
Uses face recognition to automatically mark attendance and log to CSV.
Run: python attendance.py
"""

import cv2
import face_recognition
import numpy as np
import csv
import os
from datetime import datetime, date
from face_recognizer import FaceRecognitionSystem


class AttendanceLogger:
    def __init__(self, log_dir="attendance_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.frs = FaceRecognitionSystem()
        self.today_log = os.path.join(log_dir, f"attendance_{date.today()}.csv")
        self.marked_today = self._load_today_records()

    def _load_today_records(self):
        """Load already-marked names for today to avoid duplicates."""
        if not os.path.exists(self.today_log):
            with open(self.today_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time", "Confidence"])
        else:
            with open(self.today_log, "r") as f:
                reader = csv.DictReader(f)
                return {row["Name"] for row in reader}
        return set()

    def mark(self, name, confidence):
        """Mark attendance for a recognized person."""
        if name in self.marked_today or name == "Unknown":
            return False

        now = datetime.now()
        with open(self.today_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), f"{confidence:.1f}%"])

        self.marked_today.add(name)
        print(f"  ✅ Attendance marked: {name} at {now.strftime('%H:%M:%S')}")
        return True

    def run(self, scale=0.5, tolerance=0.5):
        """Run attendance system from webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        print(f"[INFO] Attendance system started. Log: {self.today_log}")
        print("[INFO] Press 'q' to stop.")

        marked_flash = {}  # name → flash timer for green highlight
        process_frame = True
        face_locations, face_names, confidences = [], [], []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if process_frame:
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb, model="hog")
                encs = face_recognition.face_encodings(rgb, face_locations)
                face_names, confidences = [], []

                for enc in encs:
                    name, conf = self.frs._identify_face(enc, tolerance)
                    face_names.append(name)
                    confidences.append(conf)
                    if name != "Unknown":
                        newly_marked = self.mark(name, conf)
                        if newly_marked:
                            marked_flash[name] = 30  # show for 30 frames

            process_frame = not process_frame

            # Draw
            for (top, right, bottom, left), name, conf in zip(face_locations, face_names, confidences):
                top, right, bottom, left = int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)

                is_flash = name in marked_flash and marked_flash[name] > 0
                color = (0, 255, 120) if is_flash else ((0, 200, 0) if name != "Unknown" else (0, 0, 200))
                thickness = 3 if is_flash else 2

                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
                status = "✓ MARKED" if name in self.marked_today else name
                label = f"{status} ({conf:.0f}%)" if name != "Unknown" else "Unknown"
                cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, -1)
                cv2.putText(frame, label, (left + 4, bottom - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if name in marked_flash:
                    marked_flash[name] -= 1

            # Status panel
            cv2.rectangle(frame, (0, 0), (300, 55), (30, 30, 30), -1)
            cv2.putText(frame, f"Attendance: {date.today()}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"Marked today: {len(self.marked_today)} | Press Q to quit",
                        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 255, 150), 1)

            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Session ended. {len(self.marked_today)} people marked.")
        print(f"[INFO] Log saved at: {self.today_log}")
        self._print_summary()

    def _print_summary(self):
        print(f"\n{'='*40}")
        print(f"  Attendance Summary — {date.today()}")
        print(f"{'='*40}")
        with open(self.today_log, "r") as f:
            for row in csv.DictReader(f):
                print(f"  {row['Name']:<20} {row['Time']}  {row['Confidence']}")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    logger = AttendanceLogger()
    logger.run()
