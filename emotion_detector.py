"""
=========================================================
AI Face Attendance System
Emotion Detection Module — DeepFace Background Thread
Author : Malay Maity
=========================================================
"""

import threading
import queue
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False
    print("[Emotion] DeepFace not available. Emotion detection disabled.")


class EmotionDetectorThread(threading.Thread):
    """
    Runs DeepFace emotion analysis in a background daemon thread
    so it never blocks the main camera loop.

    Usage:
        det = EmotionDetectorThread()
        det.start()

        # Feed a face crop (BGR numpy array):
        det.submit(face_bgr, face_id=0)

        # Read last result:
        emotion = det.get_emotion(face_id=0)  # e.g. "happy"

        det.stop()
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._input_queue  = queue.Queue(maxsize=4)   # face crops to process
        self._results      = {}                        # face_id -> emotion str
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()
        self._available    = DEEPFACE_AVAILABLE

    @property
    def available(self):
        return self._available

    def submit(self, face_bgr: np.ndarray, face_id: int = 0):
        """Non-blocking submit. Drops frame if queue is full (backpressure)."""
        if not self._available:
            return
        try:
            self._input_queue.put_nowait((face_id, face_bgr.copy()))
        except queue.Full:
            pass  # drop — main loop must not block

    def get_emotion(self, face_id: int = 0) -> str:
        """Return last detected emotion for the given face slot, or 'Neutral'."""
        with self._lock:
            return self._results.get(face_id, "Neutral")

    def stop(self):
        self._stop_event.set()
        try:
            self._input_queue.put_nowait(None)   # unblock worker
        except queue.Full:
            pass

    def run(self):
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            face_id, face_bgr = item
            emotion = self._analyze(face_bgr)

            with self._lock:
                self._results[face_id] = emotion

    def _analyze(self, face_bgr: np.ndarray) -> str:
        if not self._available or face_bgr is None or face_bgr.size == 0:
            return "Neutral"
        try:
            result = DeepFace.analyze(
                face_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            return result.get("dominant_emotion", "Neutral").capitalize()
        except Exception:
            return "Neutral"
