import time
import threading
import sys

# Platform sound import
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    try:
        import winsound
    except ImportError:
        winsound = None
else:
    winsound = None

class DriverAlertSystem:
    """
    State-machine alert manager for driver drowsiness, yawning, and distraction.
    Includes thread-safe non-blocking audio alerts with cooldowns.
    """
    def __init__(self, 
                 sleepy_threshold=15, 
                 yawn_threshold=20, 
                 distraction_threshold=25, 
                 sound_enabled=True,
                 cooldown_sec=1.5):
        self.sleepy_threshold = sleepy_threshold
        self.yawn_threshold = yawn_threshold
        self.distraction_threshold = distraction_threshold
        self.sound_enabled = sound_enabled
        self.cooldown_sec = cooldown_sec

        # State counters
        self.sleepy_frames = 0
        self.yawn_frames = 0
        self.missing_face_frames = 0

        # Statistics
        self.total_sleep_alerts = 0
        self.total_yawn_alerts = 0
        self.total_distraction_alerts = 0

        # Cooldown timer
        self.last_sound_time = 0
        self._lock = threading.Lock()

    def update(self, detected_state, face_present=True):
        """
        Updates frame counters and evaluates the current alert status.
        detected_state: 'Normal', 'Sleepy', 'Yawning', or None
        Returns dict with alert details:
          {
             'alert_active': bool,
             'alert_type': str,  # 'NONE', 'DROWSINESS', 'FATIGUE', 'DISTRACTION'
             'message': str,
             'level': str,       # 'INFO', 'WARNING', 'DANGER'
             'color': tuple      # BGR
          }
        """
        # If face is not detected (e.g. driver turned away or left frame)
        if not face_present:
            self.missing_face_frames += 1
            self.sleepy_frames = 0
            self.yawn_frames = 0
        else:
            self.missing_face_frames = 0
            if detected_state == "Sleepy":
                self.sleepy_frames += 1
                self.yawn_frames = max(0, self.yawn_frames - 1)
            elif detected_state == "Yawning":
                self.yawn_frames += 1
                self.sleepy_frames = max(0, self.sleepy_frames - 1)
            else:  # Normal
                self.sleepy_frames = max(0, self.sleepy_frames - 2)
                self.yawn_frames = max(0, self.yawn_frames - 2)

        # Evaluate priority 1: DROWSINESS (Most critical)
        if self.sleepy_frames >= self.sleepy_threshold:
            self._trigger_sound(freq=1500, duration=350)
            self.total_sleep_alerts += 1
            return {
                "alert_active": True,
                "alert_type": "DROWSINESS",
                "message": "CRITICAL: SLEEPINESS DETECTED! WAKE UP!",
                "level": "DANGER",
                "color": (0, 0, 255) # Red
            }

        # Evaluate priority 2: DISTRACTION (Driver looking away / missing face)
        if self.missing_face_frames >= self.distraction_threshold:
            self._trigger_sound(freq=900, duration=200)
            self.total_distraction_alerts += 1
            return {
                "alert_active": True,
                "alert_type": "DISTRACTION",
                "message": "WARNING: DRIVER DISTRACTED / LOOKING AWAY!",
                "level": "WARNING",
                "color": (255, 0, 255) # Magenta
            }

        # Evaluate priority 3: FATIGUE (Yawning)
        if self.yawn_frames >= self.yawn_threshold:
            self._trigger_sound(freq=1100, duration=200)
            self.total_yawn_alerts += 1
            return {
                "alert_active": True,
                "alert_type": "FATIGUE",
                "message": "FATIGUE ALERT: FREQUENT YAWNING DETECTED!",
                "level": "WARNING",
                "color": (0, 165, 255) # Orange
            }

        # Normal condition
        return {
            "alert_active": False,
            "alert_type": "NONE",
            "message": "Driver Attentive & Normal",
            "level": "INFO",
            "color": (0, 200, 0) # Green
        }

    def _trigger_sound(self, freq=1200, duration=250):
        """Dispatches non-blocking audio alert in a background thread."""
        if not self.sound_enabled or not IS_WINDOWS or winsound is None:
            return

        now = time.time()
        with self._lock:
            if now - self.last_sound_time < self.cooldown_sec:
                return
            self.last_sound_time = now

        def _play():
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def reset(self):
        """Reset all counters."""
        self.sleepy_frames = 0
        self.yawn_frames = 0
        self.missing_face_frames = 0
