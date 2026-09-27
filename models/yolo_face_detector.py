import os
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOFaceDetector:
    """
    Robust Face Detector using YOLOv8 with automatic model downloading
    and multi-tier fallback (yolov8n-face -> yolov8n -> Haar Cascade).
    """
    YOLO_FACE_URL = "https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt"

    def __init__(self, weights_path="models/weights/yolov8n-face.pt", conf_thresh=0.45, device="cpu"):
        self.conf_thresh = conf_thresh
        self.device = device
        self.weights_path = str(weights_path)
        self.model = None
        self.detector_type = "yolo_face"

        self._initialize_detector()

    def _initialize_detector(self):
        """Attempts to load yolov8n-face, downloads if missing, or falls back to yolov8n/Haar."""
        # 1. Check if custom yolov8n-face weights exist or can be downloaded
        if not os.path.exists(self.weights_path):
            print(f"[YOLO Face] Weights not found at {self.weights_path}. Attempting download...")
            try:
                os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
                urllib.request.urlretrieve(self.YOLO_FACE_URL, self.weights_path)
                print(f"[YOLO Face] Downloaded YOLOv8 face model successfully.")
            except Exception as e:
                print(f"[YOLO Face] Direct download failed ({e}). Checking local fallback...")

        # 2. Try loading yolov8n-face
        if os.path.exists(self.weights_path):
            try:
                self.model = YOLO(self.weights_path)
                self.detector_type = "yolo_face"
                print(f"[YOLO Face] Loaded dedicated YOLOv8 Face detector from {self.weights_path}")
                return
            except Exception as e:
                print(f"[YOLO Face] Failed loading {self.weights_path}: {e}")

        # 3. Fallback to standard YOLOv8n
        try:
            print("[YOLO Face] Initializing standard YOLOv8n detector...")
            self.model = YOLO("yolov8n.pt")
            self.detector_type = "yolov8_standard"
            print("[YOLO Face] Standard YOLOv8n loaded successfully.")
            return
        except Exception as e:
            print(f"[YOLO Face] YOLOv8n loading failed: {e}")

        # 4. Final resilient fallback: OpenCV Haar Cascade
        print("[YOLO Face] Falling back to OpenCV Haar Cascade face detector.")
        self.detector_type = "haar"
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame):
        """
        Detects faces in the given frame.
        Returns a list of dicts: [{'box': (x1, y1, x2, y2), 'conf': float, 'area': int}]
        """
        h, w = frame.shape[:2]
        faces = []

        if self.detector_type in ("yolo_face", "yolov8_standard"):
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                verbose=False,
                device=self.device
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # For yolo_face, class 0 is face. For standard yolo, class 0 is person.
                    if self.detector_type == "yolo_face" or (self.detector_type == "yolov8_standard" and cls_id == 0):
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy

                        # If using standard YOLO (person box), approximate upper 35% as face/head
                        if self.detector_type == "yolov8_standard":
                            person_h = y2 - y1
                            y2 = y1 + int(person_h * 0.35)

                        # Clip coordinates to frame boundary
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y2 = max(y1 + 1, min(y2, h))
                        area = (x2 - x1) * (y2 - y1)

                        faces.append({
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "area": area
                        })

        elif self.detector_type == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, fw, fh) in detected:
                faces.append({
                    "box": (x, y, x + fw, y + fh),
                    "conf": 0.85,
                    "area": fw * fh
                })

        return faces

    def get_primary_driver_face(self, frame, padding_ratio=0.18):
        """
        Returns the primary driver's face (largest detected face area),
        cropped with generous padding for mouth/eye state analysis.
        Returns: (cropped_face_bgr, (x1, y1, x2, y2), confidence) or (None, None, 0.0)
        """
        faces = self.detect_faces(frame)
        if not faces:
            return None, None, 0.0

        # Sort by bounding box area descending - driver is closest/largest
        faces.sort(key=lambda f: f["area"], reverse=True)
        primary = faces[0]
        x1, y1, x2, y2 = primary["box"]
        conf = primary["conf"]

        h, w = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1

        # Apply padding around face to ensure eyes and open jaw (yawning) fit
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None, None, 0.0

        return face_crop, (x1, y1, x2, y2), conf
