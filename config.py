import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
WEIGHTS_DIR = BASE_DIR / "models" / "weights"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Classes
CLASSES = ["Normal", "Sleepy", "Yawning"]
NUM_CLASSES = len(CLASSES)

# Class color mapping (BGR for OpenCV)
CLASS_COLORS = {
    "Normal": (0, 200, 0),     # Vibrant Green
    "Sleepy": (0, 0, 255),     # Alert Red
    "Yawning": (0, 165, 255),  # Warning Orange
    "Distracted": (255, 0, 255) # Magenta
}

# Model Configurations
MODEL_NAME = "efficientnet_b0"
CLASSIFIER_WEIGHTS_PATH = WEIGHTS_DIR / "efficientnet_b0_driver.pth"
YOLO_FACE_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8n-face.pt"
FALLBACK_YOLO_PATH = "yolov8n.pt"

# Image Preprocessing (EfficientNet-B0 standard)
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Real-Time Monitoring & Alert Thresholds
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW_SIZE = 6  # Moving average over last N frames to eliminate transient flickers

# Sustained frame thresholds before triggering warning / alarm
SLEEPY_ALERT_FRAMES = 15       # Approx 0.5 - 0.75 seconds of sleepy state
YAWN_ALERT_FRAMES = 20         # Sustained yawning threshold
DISTRACTION_ALERT_FRAMES = 25  # Sustained absence of driver face or looking away

# Sound Alert settings
ENABLE_AUDIO_ALERT = True
ALERT_BEEP_FREQ = 1200         # Hz
ALERT_BEEP_DURATION = 250      # ms
ALERT_COOLDOWN_SECONDS = 1.5   # Prevent constant non-stop audio overlapping
