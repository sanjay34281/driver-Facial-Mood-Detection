import cv2
import numpy as np

class HUDDisplay:
    """
    Renders a modern, professional cockpit HUD overlay for the Driver Monitoring System.
    """
    def __init__(self, classes=("Normal", "Sleepy", "Yawning")):
        self.classes = classes

    def draw_hud(self, 
                 frame, 
                 face_box=None, 
                 state_probs=None, 
                 alert_info=None, 
                 fps=0.0, 
                 sleepy_progress=0.0, 
                 yawn_progress=0.0,
                 sound_enabled=True):
        """
        Draws the complete HUD overlay on the frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Semi-transparent Top Header Bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Header Title
        cv2.putText(canvas, "DRIVER MONITORING SYSTEM", (20, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, "YOLOv8 Face + EfficientNet-B0", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

        # FPS & Audio Indicator in Top Right
        fps_text = f"FPS: {fps:04.1f}"
        sound_text = "AUDIO: [ON]" if sound_enabled else "AUDIO: [OFF]"
        cv2.putText(canvas, fps_text, (w - 130, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, sound_text, (w - 130, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200) if sound_enabled else (100, 100, 255), 1, cv2.LINE_AA)

        # 2. Main Alert / Status Banner below header
        if alert_info:
            color = alert_info["color"]
            msg = alert_info["message"]
            active = alert_info["alert_active"]

            # Draw alert bar
            bar_y1 = 60
            bar_y2 = 95
            overlay = canvas.copy()
            bg_color = (0, 0, 140) if alert_info["level"] == "DANGER" else (25, 25, 30)
            cv2.rectangle(overlay, (15, bar_y1), (w - 15, bar_y2), bg_color, -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

            # Border
            cv2.rectangle(canvas, (15, bar_y1), (w - 15, bar_y2), color, 2 if active else 1)

            # Pulsing text if critical alert active
            cv2.putText(canvas, msg, (35, bar_y1 + 24),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, color if not active else (255, 255, 255), 2 if active else 1, cv2.LINE_AA)

        # 3. Face Bounding Box & Target Brackets
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            box_color = alert_info["color"] if alert_info else (0, 255, 0)
            
            # Corner accents (sleek targeting aesthetic)
            self._draw_corner_rect(canvas, (x1, y1, x2, y2), box_color, corner_len=18, thickness=2)

            # Face label
            if state_probs is not None:
                max_cls = max(state_probs, key=state_probs.get)
                max_conf = state_probs[max_cls]
                label = f"{max_cls} ({max_conf * 100:.1f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                
                # Tag background
                tag_y1 = max(0, y1 - 25)
                cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(canvas, label, (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. Probabilities & Telemetry Panel (Bottom-Left)
        if state_probs is not None:
            panel_w = 210
            panel_h = 135
            panel_x = 20
            panel_y = h - panel_h - 45

            overlay = canvas.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 90), 1)

            cv2.putText(canvas, "STATE CONFIDENCE", (panel_x + 10, panel_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            line_y = panel_y + 45
            bar_max_w = 90
            for cls_name in self.classes:
                prob = state_probs.get(cls_name, 0.0)
                # Label
                cv2.putText(canvas, f"{cls_name[:6]}:", (panel_x + 10, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                
                # Bar background
                cv2.rectangle(canvas, (panel_x + 65, line_y - 10), (panel_x + 65 + bar_max_w, line_y), (50, 50, 60), -1)
                
                # Filled bar
                fill_w = int(bar_max_w * prob)
                bar_col = (0, 220, 0) if cls_name == "Normal" else ((0, 0, 240) if cls_name == "Sleepy" else (0, 165, 255))
                if fill_w > 0:
                    cv2.rectangle(canvas, (panel_x + 65, line_y - 10), (panel_x + 65 + fill_w, line_y), bar_col, -1)
                
                # Percentage text
                cv2.putText(canvas, f"{prob * 100:3.0f}%", (panel_x + 162, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

                line_y += 28

        # 5. Drowsiness / Yawn Stress Meter (Bottom-Right)
        meter_w = 170
        meter_h = 75
        meter_x = w - meter_w - 20
        meter_y = h - meter_h - 45

        overlay = canvas.copy()
        cv2.rectangle(overlay, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 80, 90), 1)

        cv2.putText(canvas, "RISK INDICATORS", (meter_x + 10, meter_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # Sleep progress bar
        cv2.putText(canvas, "Sleep:", (meter_x + 10, meter_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.rectangle(canvas, (meter_x + 55, meter_y + 28), (meter_x + 155, meter_y + 38), (50, 50, 60), -1)
        s_fill = int(100 * min(1.0, sleepy_progress))
        if s_fill > 0:
            cv2.rectangle(canvas, (meter_x + 55, meter_y + 28), (meter_x + 55 + s_fill, meter_y + 38), (0, 0, 255), -1)

        # Yawn progress bar
        cv2.putText(canvas, "Yawn:", (meter_x + 10, meter_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.rectangle(canvas, (meter_x + 55, meter_y + 48), (meter_x + 155, meter_y + 58), (50, 50, 60), -1)
        y_fill = int(100 * min(1.0, yawn_progress))
        if y_fill > 0:
            cv2.rectangle(canvas, (meter_x + 55, meter_y + 48), (meter_x + 55 + y_fill, meter_y + 58), (0, 165, 255), -1)

        # 6. Bottom Controls Bar
        cv2.rectangle(canvas, (0, h - 25), (w, h), (10, 10, 15), -1)
        cv2.putText(canvas, "[Q]: Quit  |  [S]: Audio Mute/Unmute  |  [R]: Reset Alerts  |  [C]: Snapshot",
                    (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        return canvas

    def _draw_corner_rect(self, img, bbox, color, corner_len=15, thickness=2):
        """Draws aesthetic brackets at the corners of a bounding box."""
        x1, y1, x2, y2 = bbox
        # Top-left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)
        # Subtle rectangle box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
