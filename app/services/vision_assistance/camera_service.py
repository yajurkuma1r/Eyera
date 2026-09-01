import cv2
import numpy as np
from typing import Optional


class CameraService:
    """
    Live Camera Service for Eyera Smart Glasses.
    Accesses the real-time webcam video stream and captures live frames on demand.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap = None

    def _get_capture(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self._cap

    def is_available(self) -> bool:
        """
        Tests if a real live camera device is connected and accessible.
        """
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return bool(ret and frame is not None)
        except Exception:
            return False

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Captures the CURRENT live camera frame.
        Flushes buffered frames so the captured image reflects the exact live moment.
        """
        try:
            cap = self._get_capture()
            if not cap.isOpened():
                print("[CAMERA] Warning: Unable to open camera device.")
                return None

            # Flush hardware buffer to guarantee instant real-time frame
            for _ in range(2):
                cap.grab()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CAMERA] Warning: Failed to retrieve live frame.")
                return None

            return frame
        except Exception as e:
            print(f"[CAMERA] Frame capture error: {e}")
            return None

    def release(self):
        """
        Releases the camera hardware stream.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
