import os
import sys
import time
from collections import deque
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from models.efficientnet_classifier import get_classifier
from models.yolo_face_detector import YOLOFaceDetector
from src.alert_system import DriverAlertSystem
from src.hud_display import HUDDisplay
from src.dataset_utils import get_data_transforms

def run_realtime_detection(source=config.CAMERA_INDEX, use_gpu=True):
    print("=" * 70)
    print("   REAL-TIME DRIVER MOOD & DISTRACTION MONITORING SYSTEM")
    print("   Powered by YOLOv8 Face Detection & EfficientNet-B0 CNN")
    print("=" * 70)

    # 1. Device Setup
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[Engine] Inference Device: {device}")
    if device.type == "cuda":
        print(f"[Engine] GPU Acceleration: {torch.cuda.get_device_name(0)}")

    # 2. Load Models
    print("[Engine] Initializing YOLOv8 Face Detector...")
    face_detector = YOLOFaceDetector(
        weights_path=config.YOLO_FACE_WEIGHTS_PATH,
        conf_thresh=0.45,
        device=str(device)
    )

    print("[Engine] Initializing EfficientNet-B0 Classifier...")
    classifier = get_classifier(
        num_classes=config.NUM_CLASSES,
        weights_path=config.CLASSIFIER_WEIGHTS_PATH,
        device=device,
        pretrained=True
    )

    # 3. Setup Transform and Auxiliaries
    _, eval_transforms = get_data_transforms()
    alert_system = DriverAlertSystem(
        sleepy_threshold=config.SLEEPY_ALERT_FRAMES,
        yawn_threshold=config.YAWN_ALERT_FRAMES,
        distraction_threshold=config.DISTRACTION_ALERT_FRAMES,
        sound_enabled=config.ENABLE_AUDIO_ALERT
    )
    hud = HUDDisplay(classes=config.CLASSES)

    # Smoothing queue for probability distributions
    smooth_window = deque(maxlen=config.SMOOTHING_WINDOW_SIZE)

    # Snapshots folder
    snapshot_dir = config.BASE_DIR / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    # 4. Open Video Stream
    print(f"[Webcam] Opening camera index {source}...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Failed to open video source: {source}")
        return

    # Frame timing
    prev_time = time.time()
    fps = 0.0

    print("\n[Status] System Operational! Press 'q' in video window to exit.")
    print("[Hotkeys] [Q] Quit | [S] Toggle Sound | [R] Reset Alerts | [C] Capture Snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Video stream ended or frame drop detected.")
            break

        # Mirror frame for natural driver perspective
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        # 1. Detect Primary Driver Face via YOLOv8
        face_crop, face_box, face_conf = face_detector.get_primary_driver_face(frame)

        detected_state = "Normal"
        state_probs_dict = {c: 0.0 for c in config.CLASSES}
        face_present = face_crop is not None

        if face_present:
            # 2. Preprocess Face Crop for EfficientNet-B0
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)
            tensor_img = eval_transforms(pil_img).unsqueeze(0).to(device)

            # 3. Classify Driver State
            with torch.no_grad():
                logits = classifier(tensor_img)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Append to smoothing buffer
            smooth_window.append(probs)
            avg_probs = np.mean(smooth_window, axis=0)

            # Form state probabilities dictionary
            for idx, c in enumerate(config.CLASSES):
                state_probs_dict[c] = float(avg_probs[idx])

            # Predicted State
            pred_idx = int(np.argmax(avg_probs))
            detected_state = config.CLASSES[pred_idx]
        else:
            # Clear smoothing queue when face is lost
            smooth_window.clear()
            detected_state = None

        # 4. Update Alert Logic
        alert_info = alert_system.update(detected_state, face_present=face_present)

        # 5. Calculate Risk Progress Ratios
        sleepy_prog = alert_system.sleepy_frames / float(config.SLEEPY_ALERT_FRAMES)
        yawn_prog = alert_system.yawn_frames / float(config.YAWN_ALERT_FRAMES)

        # 6. Render Cockpit HUD
        display_frame = hud.draw_hud(
            frame=frame,
            face_box=face_box,
            state_probs=state_probs_dict if face_present else None,
            alert_info=alert_info,
            fps=fps,
            sleepy_progress=sleepy_prog,
            yawn_progress=yawn_prog,
            sound_enabled=alert_system.sound_enabled
        )

        # 7. Show Real-time Video Window
        cv2.imshow("Driver Monitoring System", display_frame)

        # 8. Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[User] Exit commanded.")
            break
        elif key == ord('s'):
            alert_system.sound_enabled = not alert_system.sound_enabled
            status = "ENABLED" if alert_system.sound_enabled else "MUTED"
            print(f"[Settings] Audio Alerts: {status}")
        elif key == ord('r'):
            alert_system.reset()
            print("[Alerts] Counters manually reset.")
        elif key == ord('c'):
            snap_path = snapshot_dir / f"driver_event_{int(time.time())}.jpg"
            cv2.imwrite(str(snap_path), display_frame)
            print(f"[Snapshot] Saved to {snap_path.name}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Shutdown] Monitoring system stopped successfully.")

if __name__ == "__main__":
    run_realtime_detection()
