"""
=========================================================
AI Face Attendance System
Liveness Detection Module — OpenCV Eye Cascade Blink Detection
Author : Malay Maity
=========================================================

Uses the Haar eye cascade (bundled with opencv-contrib-python) to
detect whether eyes are open or closed on each frame.

Blink logic:
  - Eyes visible   → "open" state
  - Eyes invisible (while face IS visible) → "closed" state
  - closed → open transition counts as ONE blink

This approach requires NO extra packages beyond OpenCV.
"""

import cv2
import numpy as np
import os


# OpenCV ships these cascades inside the package
_EYE_CASCADE_PATH  = os.path.join(cv2.data.haarcascades,
                                  "haarcascade_eye.xml")
_FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades,
                                  "haarcascade_frontalface_default.xml")


class LivenessDetector:
    """
    Blink-based liveness detector using OpenCV Haar cascades.

    Usage:
        detector = LivenessDetector(required_blinks=1)
        result   = detector.process_frame(frame)
        # result: {"verified": bool, "blinks": int, "ear": float, "status": str}
    """

    def __init__(self, ear_threshold=0.21, consec_frames=3, required_blinks=1):
        # ear_threshold / consec_frames kept for API compatibility but unused
        self.required_blinks = required_blinks

        self._blink_count   = 0
        self._eyes_open     = True    # last known state
        self._verified      = False
        self._available     = os.path.exists(_EYE_CASCADE_PATH)

        if self._available:
            self._eye_cascade  = cv2.CascadeClassifier(_EYE_CASCADE_PATH)
            self._face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
        else:
            print("[Liveness] Eye cascade XML not found. Liveness detection disabled.")

    @property
    def available(self):
        return self._available

    def reset(self):
        """Reset blink counter (call when a new session starts)."""
        self._blink_count = 0
        self._eyes_open   = True
        self._verified    = False

    def process_frame(self, frame):
        """
        Process a BGR frame and update liveness state.

        Returns dict:
            verified : bool   — True once enough blinks detected
            blinks   : int    — blink count so far
            ear      : float  — 1.0 if eyes open, 0.0 if closed (proxy)
            status   : str    — human-readable status string
        """
        if not self._available:
            return {
                "verified": True,
                "blinks":   0,
                "ear":      1.0,
                "status":   "Liveness N/A"
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face first (restrict eye search to face ROI for speed + accuracy)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        eyes_detected = False

        if len(faces) > 0:
            # Use the largest face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            # Only search top half of face for eyes
            roi = gray[y: y + h // 2, x: x + w]
            eyes = self._eye_cascade.detectMultiScale(
                roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            eyes_detected = len(eyes) >= 1

            # Blink = was open, now closed
            if not eyes_detected and self._eyes_open:
                pass  # start of close
            elif eyes_detected and not self._eyes_open:
                # Eyes re-opened → blink complete
                self._blink_count += 1

            self._eyes_open = eyes_detected
        else:
            # No face → can't judge
            self._eyes_open = True

        if self._blink_count >= self.required_blinks:
            self._verified = True

        ear_proxy = 1.0 if eyes_detected else 0.0

        if self._verified:
            status = f"LIVENESS: VERIFIED ({self._blink_count} blink)"
        else:
            status = f"BLINK TO VERIFY ({self._blink_count}/{self.required_blinks})"

        return {
            "verified": self._verified,
            "blinks":   self._blink_count,
            "ear":      ear_proxy,
            "status":   status
        }

    def draw_debug(self, frame, result):
        """Draw liveness status on the frame (top-left corner)."""
        color = (0, 220, 0) if result["verified"] else (0, 80, 255)
        cv2.putText(frame, result["status"], (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        eye_state = "Eyes: OPEN" if result["ear"] > 0.5 else "Eyes: CLOSED"
        cv2.putText(frame, eye_state, (10, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
