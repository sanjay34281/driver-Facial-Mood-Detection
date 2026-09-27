import os
import sys
import time
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

import config
from models.efficientnet_classifier import get_classifier
from models.yolo_face_detector import YOLOFaceDetector
from src.hud_display import HUDDisplay
from src.dataset_utils import get_data_transforms

st.set_page_config(
    page_title="Driver Mood & Distraction Detection",
    page_icon="🚗",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚗 Driver Mood & Distraction Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-Time Deep Learning Monitoring with YOLOv8 & EfficientNet-B0</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration & Controls")
mode = st.sidebar.selectbox("Select Mode", ["Live Webcam Monitoring", "Upload Image / Test Sample", "Model Architecture & Info"])
conf_thresh = st.sidebar.slider("YOLOv8 Face Confidence", 0.20, 0.90, 0.45, 0.05)
device_choice = st.sidebar.selectbox("Computation Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])

# Caching Models
@st.cache_resource
def load_models(dev):
    device = torch.device(dev)
    detector = YOLOFaceDetector(weights_path=config.YOLO_FACE_WEIGHTS_PATH, conf_thresh=conf_thresh, device=str(device))
    classifier = get_classifier(
        num_classes=config.NUM_CLASSES,
        weights_path=config.CLASSIFIER_WEIGHTS_PATH,
        device=device,
        pretrained=True
    )
    _, transforms = get_data_transforms()
    hud = HUDDisplay(classes=config.CLASSES)
    return detector, classifier, transforms, hud

detector, classifier, transforms, hud = load_models(device_choice)

if mode == "Upload Image / Test Sample":
    st.subheader("📸 Image Analysis Mode")
    uploaded_file = st.file_uploader("Upload an image of a driver", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1.2, 1])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Run Face Detection
        face_crop, face_box, face_conf = detector.get_primary_driver_face(frame)

        if face_crop is not None:
            # Classification
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(rgb_face)
            tensor_img = transforms(crop_pil).unsqueeze(0).to(device_choice)

            with torch.no_grad():
                logits = classifier(tensor_img)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            probs_dict = {config.CLASSES[i]: float(probs[i]) for i in range(len(config.CLASSES))}
            pred_class = config.CLASSES[np.argmax(probs)]
            pred_conf = float(np.max(probs))

            alert_info = {
                "alert_active": pred_class != "Normal",
                "alert_type": pred_class.upper(),
                "message": f"Detected: {pred_class} ({pred_conf*100:.1f}%)",
                "level": "DANGER" if pred_class == "Sleepy" else ("WARNING" if pred_class == "Yawning" else "INFO"),
                "color": config.CLASS_COLORS[pred_class]
            }

            annotated = hud.draw_hud(frame, face_box, probs_dict, alert_info, fps=0.0)

            with col1:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analyzed Driver Frame", use_container_width=True)

            with col2:
                st.markdown("### Telemetry Results")
                st.metric("Primary Classification", pred_class, f"{pred_conf*100:.1f}% Confidence")
                st.metric("Face Detection Confidence", f"{face_conf*100:.1f}%")

                st.markdown("#### Class Probabilities")
                for cls_name, p in probs_dict.items():
                    st.write(f"**{cls_name}**")
                    st.progress(float(p))
                    st.caption(f"{p*100:.2f}%")
        else:
            st.warning("No face detected in this image. Ensure good lighting and frontal face view.")
            st.image(pil_img, caption="Original Image", use_container_width=True)

elif mode == "Live Webcam Monitoring":
    st.subheader("📹 Live Driver Monitoring")
    st.info("Click 'Start Camera' to initialize the video feed. To run native desktop performance at high FPS with audio alerts, execute `python detect_realtime.py` in your terminal.")

    run_cam = st.toggle("Start Camera Feed")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        prev_time = time.time()
        fps = 0.0

        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
            prev_time = curr_time

            face_crop, face_box, face_conf = detector.get_primary_driver_face(frame)
            probs_dict = {c: 0.0 for c in config.CLASSES}
            pred_class = "Normal"

            if face_crop is not None:
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(rgb_face)
                tensor_img = transforms(crop_pil).unsqueeze(0).to(device_choice)

                with torch.no_grad():
                    logits = classifier(tensor_img)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                probs_dict = {config.CLASSES[i]: float(probs[i]) for i in range(len(config.CLASSES))}
                pred_class = config.CLASSES[np.argmax(probs)]
                alert_info = {
                    "alert_active": pred_class != "Normal",
                    "alert_type": pred_class.upper(),
                    "message": f"Status: {pred_class}",
                    "level": "DANGER" if pred_class == "Sleepy" else "INFO",
                    "color": config.CLASS_COLORS[pred_class]
                }
            else:
                alert_info = {
                    "alert_active": True,
                    "alert_type": "DISTRACTION",
                    "message": "DISTRACTION: DRIVER MISSING",
                    "level": "WARNING",
                    "color": (255, 0, 255)
                }

            display_frame = hud.draw_hud(frame, face_box, probs_dict if face_crop is not None else None, alert_info, fps=fps)
            frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

elif mode == "Model Architecture & Info":
    st.subheader("🧠 System Architecture & Methodology")
    st.markdown("""
    ### Project Overview
    This real-time Driver Monitoring System (DMS) is designed to mitigate motor vehicle accidents caused by driver fatigue, micro-sleeps, and distraction.

    #### Pipeline Components:
    1. **Real-time Video Feed**: Captures webcam stream at 30+ FPS via OpenCV.
    2. **YOLOv8 Face Detection**: Employs deep convolutional object detection to localize the driver's face, accommodating head turns and varying light conditions.
    3. **EfficientNet-B0 Classifier**:
       - Uses transfer learning on ImageNet-pretrained weights.
       - Replaces classification head with custom Linear/Dropout/SiLU layers.
       - Outputs class probabilities for **Normal**, **Sleepy** (eye closure / nodding), and **Yawning** (mouth opening / fatigue).
    4. **Cockpit HUD & Multi-Tier Alerts**:
       - Non-blocking audio beeps with cooldown management.
       - Telemetry overlay showing live confidence meters and risk indicators.
    """)
