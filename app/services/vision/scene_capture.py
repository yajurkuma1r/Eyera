import cv2
import numpy as np
from typing import Any, Dict, Optional, List


class ScenePreset:
    """
    Standard test scenes for offline development, demonstrations, and automated testing.
    """
    CAFE_MENU = {
        "scene_name": "Cafe Counter & Menu",
        "ocr_text": "DAILY SPECIALS\nEspresso: $3.50\nCaffe Latte: $4.50\nCappuccino: $4.00\nIced Americano: $3.75\nFresh Croissant: $2.50\nGluten-Free Muffin: $3.00",
        "detected_objects": [
            {"label": "dining table", "position": "center", "distance": "1.0m", "confidence": 0.92},
            {"label": "cup", "position": "center-left", "distance": "0.7m", "confidence": 0.89},
            {"label": "person", "position": "center-right", "distance": "1.8m", "confidence": 0.94}
        ],
        "warnings": []
    }

    STREET_CROSSING = {
        "scene_name": "Street Intersection",
        "ocr_text": "PEDESTRIAN CROSSING\nMAIN ST & 4TH AVE",
        "detected_objects": [
            {"label": "traffic light", "position": "center", "distance": "6.0m", "confidence": 0.95},
            {"label": "car", "position": "left", "distance": "3.5m", "confidence": 0.91},
            {"label": "pole", "position": "right", "distance": "2.0m", "confidence": 0.88}
        ],
        "warnings": [
            {"type": "approaching", "object": "Car", "message": "Car approaching on the left", "depth": 380.0}
        ]
    }

    OFFICE_ROOM = {
        "scene_name": "Office Room Entrance",
        "ocr_text": "ROOM 302 - INNOVATION LAB\nAUTHORIZED ACCESS ONLY",
        "detected_objects": [
            {"label": "door", "position": "center", "distance": "2.5m", "confidence": 0.90},
            {"label": "chair", "position": "left", "distance": "1.2m", "confidence": 0.85},
            {"label": "table", "position": "center-left", "distance": "1.5m", "confidence": 0.87}
        ],
        "warnings": []
    }

    BUS_STOP = {
        "scene_name": "Bus Transit Stop",
        "ocr_text": "ROUTE 42 EXPRESS\nNEXT BUS: 5 MINS\nDESTINATION: DOWNTOWN TRANSIT CENTER",
        "detected_objects": [
            {"label": "bus", "position": "left ahead", "distance": "8.0m", "confidence": 0.93},
            {"label": "bench", "position": "right", "distance": "1.5m", "confidence": 0.89},
            {"label": "person", "position": "right", "distance": "2.1m", "confidence": 0.91}
        ],
        "warnings": []
    }


class SceneCaptureService:
    """
    Service responsible for capturing visual and OCR data from the camera or presets.
    """

    PRESETS = {
        "menu": ScenePreset.CAFE_MENU,
        "street": ScenePreset.STREET_CROSSING,
        "office": ScenePreset.OFFICE_ROOM,
        "bus": ScenePreset.BUS_STOP
    }

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._vision_service = None

    def get_preset(self, name: str) -> Dict[str, Any]:
        """
        Retrieves a predefined simulated visual scene.
        """
        return self.PRESETS.get(name.lower(), ScenePreset.CAFE_MENU)

    def list_presets(self) -> List[str]:
        """
        Lists available preset names.
        """
        return list(self.PRESETS.keys())

    def capture_live(self) -> Dict[str, Any]:
        """
        Captures a live frame from the webcam and runs object detection and OCR.
        Falls back gracefully to a mock scene if no camera is connected.
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[SceneCapture] Warning: Camera not accessible. Falling back to Cafe Menu preset.")
            return ScenePreset.CAFE_MENU

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("[SceneCapture] Warning: Could not read frame from camera. Using preset.")
            return ScenePreset.CAFE_MENU

        # Attempt to run OCR
        ocr_text = self._extract_ocr_text(frame)

        # Attempt to run object detection
        detected_objects, warnings = self._extract_objects_and_warnings(frame)

        return {
            "scene_name": "Live Camera Snapshot",
            "ocr_text": ocr_text,
            "detected_objects": detected_objects,
            "warnings": warnings
        }

    def _extract_ocr_text(self, frame: np.ndarray) -> str:
        """
        Performs OCR on the captured image using available libraries.
        """
        try:
            import pytesseract
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Basic preprocessing: thresholding for contrast
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(thresh)
            return text.strip()
        except Exception:
            # Fallback if tesseract binary is not installed locally
            return ""

    def _extract_objects_and_warnings(self, frame: np.ndarray):
        """
        Uses YOLO to detect objects and categorize positions.
        """
        objects = []
        warnings = []
        try:
            from ultralytics import YOLO
            yolo = YOLO("yolov8n.pt")
            results = yolo(frame, verbose=False)[0]

            h, w = frame.shape[:2]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = yolo.names[cls_id]
                conf = float(box.conf[0])
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                
                # Determine spatial position
                if cx < w * 0.35:
                    position = "left"
                elif cx > w * 0.65:
                    position = "right"
                else:
                    position = "center"

                # Approximate distance by bounding box height ratio
                box_h_ratio = (y2 - y1) / h
                if box_h_ratio > 0.6:
                    distance = "0.5m - 1.0m (Very Close)"
                    warnings.append({
                        "type": "proximity",
                        "object": label,
                        "message": f"{label} is very close ahead"
                    })
                elif box_h_ratio > 0.3:
                    distance = "1.0m - 2.0m (Ahead)"
                else:
                    distance = "> 2.0m (Further away)"

                objects.append({
                    "label": label,
                    "position": position,
                    "distance": distance,
                    "confidence": round(conf, 2)
                })

        except Exception as e:
            print(f"[SceneCapture] YOLO inference note: {e}")

        return objects, warnings
