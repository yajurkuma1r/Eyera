import cv2
import numpy as np
from typing import Optional


class CameraService:
    """
    Live Camera Service for Eyera Smart Glasses.
    Manages live webcam hardware access and captures real-time video frames on demand.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap = None

    def _get_capture(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
            # Set resolution if supported
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self._cap

    def is_available(self) -> bool:
        """
        Tests if the camera can be opened and read.
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
        Flushes the internal hardware buffer to ensure the frame is the exact live moment.
        """
        try:
            cap = self._get_capture()
            if not cap.isOpened():
                print("[CameraService] Warning: Could not open video device.")
                return None

            # Flush buffer (grab 2 frames) so we get the fresh real-time frame
            for _ in range(2):
                cap.grab()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CameraService] Warning: Failed to retrieve frame from camera.")
                return None

            return frame
        except Exception as e:
            print(f"[CameraService] Camera capture error: {e}")
            return None

    def release(self):
        """
        Releases the camera resource.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
