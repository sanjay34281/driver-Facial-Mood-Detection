import os
import cv2
import time
import random
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from models.yolo_face_detector import YOLOFaceDetector

def run_data_collector():
    print("=" * 65)
    print("      DRIVER MONITORING SYSTEM - DATASET COLLECTOR")
    print("=" * 65)
    print("Controls:")
    print("  [1] Select 'Normal' class")
    print("  [2] Select 'Sleepy' class (close eyes or look drowsy)")
    print("  [3] Select 'Yawning' class (open mouth widely)")
    print("  [SPACE] Capture single face crop for current class")
    print("  [B] Burst Capture: Automatically take 30 consecutive crops")
    print("  [Q] Exit collector")
    print("=" * 65)

    detector = YOLOFaceDetector(weights_path=config.YOLO_FACE_WEIGHTS_PATH)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("[Error] Could not open webcam index", config.CAMERA_INDEX)
        return

    current_class_idx = 0
    burst_remaining = 0
    burst_class = None

    # Count existing images
    def count_images():
        counts = {}
        for c in config.CLASSES:
            tr_cnt = len(list((config.TRAIN_DIR / c).glob("*.jpg")))
            val_cnt = len(list((config.VAL_DIR / c).glob("*.jpg")))
            counts[c] = (tr_cnt, val_cnt)
        return counts

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        current_class = config.CLASSES[current_class_idx]

        face_crop, face_box, conf = detector.get_primary_driver_face(frame)

        # Draw Face Box
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Handle burst capture
        if burst_remaining > 0 and face_crop is not None:
            save_crop(face_crop, burst_class)
            burst_remaining -= 1
            time.sleep(0.04)

        # Top Information Overlay
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        active_str = f"ACTIVE TARGET: [{current_class.upper()}]"
        cv2.putText(frame, active_str, (15, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, config.CLASS_COLORS[current_class], 2)

        # Display Counts
        counts = count_images()
        info_x = 350
        for c in config.CLASSES:
            tr, vl = counts[c]
            txt = f"{c}: {tr+vl}"
            cv2.putText(frame, txt, (info_x, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.CLASS_COLORS[c], 1)
            info_x += 100

        # Burst Banner
        if burst_remaining > 0:
            cv2.putText(frame, f"BURST CAPTURING: {burst_remaining} left...", (15, 90),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

        # Instructions Bar
        cv2.rectangle(frame, (0, h - 25), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "[1]: Normal | [2]: Sleepy | [3]: Yawn | [SPACE]: Save 1 | [B]: Burst 30 | [Q]: Quit",
                    (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Driver Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            current_class_idx = 0
            print(f"[Selected] Class: {config.CLASSES[0]}")
        elif key == ord('2'):
            current_class_idx = 1
            print(f"[Selected] Class: {config.CLASSES[1]}")
        elif key == ord('3'):
            current_class_idx = 2
            print(f"[Selected] Class: {config.CLASSES[2]}")
        elif key == 32:  # SPACE
            if face_crop is not None:
                path = save_crop(face_crop, current_class)
                print(f"[Saved] Single crop for '{current_class}' -> {path.name}")
            else:
                print("[Warning] No face detected to save!")
        elif key == ord('b'):  # Burst
            burst_class = current_class
            burst_remaining = 30
            print(f"[Burst] Starting 30-frame capture for '{burst_class}'...")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Data collection session ended.")

def save_crop(crop_img, class_name):
    # 85% train, 15% val split
    split_dir = config.TRAIN_DIR if random.random() < 0.85 else config.VAL_DIR
    target_folder = split_dir / class_name
    target_folder.mkdir(parents=True, exist_ok=True)

    filename = f"{class_name.lower()}_{int(time.time() * 1000)}_{random.randint(100, 999)}.jpg"
    target_path = target_folder / filename
    
    # Resize to standard size before saving
    resized = cv2.resize(crop_img, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target_path), resized)
    return target_path

if __name__ == "__main__":
    run_data_collector()
