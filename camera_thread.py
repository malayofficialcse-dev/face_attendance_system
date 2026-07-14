"""
=========================================================
AI Face Attendance System
Camera Thread Module — Multi-Camera Support
Author : Malay Maity
=========================================================
"""

import threading
import cv2
import time
import sys


def _open_camera(index: int, width: int, height: int):
    """
    Try to open camera with DirectShow first (Windows), then fallback to default.
    Retries the test-read several times to allow for camera warm-up lag.
    """
    backends = []
    if sys.platform == "win32":
        backends.append(cv2.CAP_DSHOW)   # DirectShow — most stable on Windows
    backends.append(cv2.CAP_ANY)          # OS default as fallback

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Retry test-read: cameras often need a few attempts after opening
        for attempt in range(10):
            ret, _ = cap.read()
            if ret:
                backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "Default"
                print(f"[Camera {index}] Opened with {backend_name} backend "
                      f"(attempt {attempt + 1})")
                return cap
            time.sleep(0.1)

        cap.release()

    return None


class CameraThread(threading.Thread):
    """
    Reads frames from a single camera in a background thread.
    Provides non-blocking get_frame() for the main loop.

    Usage:
        cam = CameraThread(camera_index=0, width=640, height=480)
        cam.start()

        frame = cam.get_frame()   # returns latest BGR frame or None

        cam.stop()
    """

    def __init__(self, camera_index: int, width: int = 640, height: int = 480):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width        = width
        self.height       = height

        self._frame       = None
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._opened      = False
        self._error       = None

    @property
    def is_opened(self) -> bool:
        return self._opened

    @property
    def error(self):
        return self._error

    def get_frame(self):
        """Returns the most recent frame (BGR ndarray) or None. Thread-safe."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = _open_camera(self.camera_index, self.width, self.height)
        if cap is None:
            self._error  = f"Cannot open camera {self.camera_index}"
            self._opened = False
            print(f"[Camera {self.camera_index}] {self._error}")
            return

        self._opened = True
        print(f"[Camera {self.camera_index}] Started successfully")

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Camera dropped — try to reopen after a short pause
                cap.release()
                time.sleep(1.0)
                cap = _open_camera(self.camera_index, self.width, self.height)
                if cap is None:
                    self._opened = False
                    break
                continue

            frame = cv2.flip(frame, 1)   # mirror

            with self._lock:
                self._frame = frame

        if cap is not None:
            cap.release()
        print(f"[Camera {self.camera_index}] Stopped")
