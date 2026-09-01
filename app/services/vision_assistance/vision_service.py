import cv2
import torch
import numpy as np
from typing import Any, Dict, List, Optional
import pytesseract
from ultralytics import YOLO
import supervision as sv


class VisionService:
    """
    Live Vision Understanding & OCR Service for Eyera.
    Processes live camera frames using YOLO (object detection),
    MiDaS (depth estimation), ByteTrack (tracking), and Tesseract (OCR).
    Executes models selectively based on required capabilities.
    """

    def __init__(self, tesseract_path: Optional[str] = None):
        print("[VISION] Initializing YOLO and Vision models...")
        # Load YOLO model
        self.yolo = YOLO("yolov8n.pt")

        # Load ByteTrack
        self.tracker = sv.ByteTrack()

        # Load MiDaS depth model
        self.midas = None
        self.transform = None
        self._init_midas()

        # Set up Tesseract if custom path or standard location
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def _init_midas(self):
        try:
            print("[VISION] Loading MiDaS depth estimation model...")
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False)
            self.midas.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
            self.transform = transforms.small_transform
        except Exception as e:
            print(f"[VISION] MiDaS load notice ({e}). Using bounding box depth fallback.")
            self.midas = None

    def _get_position(self, center_x: float, frame_width: int) -> str:
        if center_x < frame_width / 3:
            return "left"
        elif center_x < (frame_width / 3) * 2:
            return "center"
        else:
            return "right"

    def _get_depth_label(self, depth_value: float) -> str:
        if depth_value > 800:
            return "very close (< 1m)"
        elif depth_value > 400:
            return "close (1m - 2m)"
        else:
            return "far (> 2m)"

    def _get_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.midas is None or self.transform is None:
            return None
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb)
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            return prediction.cpu().numpy()
        except Exception:
            return None

    def read_text(self, frame: np.ndarray) -> str:
        """
        Runs OCR on the current camera frame to extract whatever text is physically visible.
        Returns the actual extracted text, or an empty string if no readable text is found.
        """
        if frame is None:
            return ""

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Contrast thresholding for text extraction
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            raw_text = pytesseract.image_to_string(processed)

            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            cleaned_text = " ".join(lines)

            # Fallback to standard grayscale if threshold produced no text
            if len(cleaned_text) < 3:
                raw_text2 = pytesseract.image_to_string(gray)
                lines2 = [line.strip() for line in raw_text2.split("\n") if line.strip()]
                cleaned_text = " ".join(lines2)

            if len(cleaned_text) < 3:
                return ""

            # Check alphanumeric ratio to reject OCR texture noise
            letter_count = sum(c.isalnum() for c in cleaned_text)
            if letter_count / max(len(cleaned_text), 1) < 0.4:
                return ""

            return cleaned_text.strip()
        except Exception as e:
            print(f"[OCR] Processing notice: {e}")
            return ""

    def get_scene_objects(self, frame: np.ndarray, need_depth: bool = True) -> List[Dict[str, Any]]:
        """
        Runs YOLO object detection and spatial position/depth calculation on the live camera frame.
        """
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        depth_map = self._get_depth_map(frame) if need_depth else None

        result = self.yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)

        scene_objects = []
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.5
                if conf < 0.35:
                    continue

                x1, y1, x2, y2 = detections.xyxy[i]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                class_id = int(detections.class_id[i])
                object_name = self.yolo.names[class_id]

                position = self._get_position(center_x, frame_width)

                if depth_map is not None:
                    cy_clamped = max(0, min(center_y, depth_map.shape[0] - 1))
                    cx_clamped = max(0, min(center_x, depth_map.shape[1] - 1))
                    depth_val = float(depth_map[cy_clamped, cx_clamped])
                    distance_str = self._get_depth_label(depth_val)
                else:
                    box_h_ratio = (y2 - y1) / frame_height
                    if box_h_ratio > 0.5:
                        distance_str = "very close (< 1m)"
                    elif box_h_ratio > 0.25:
                        distance_str = "close (1m - 2m)"
                    else:
                        distance_str = "far (> 2m)"

                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else i + 1

                scene_objects.append({
                    "id": track_id,
                    "label": object_name,
                    "name": object_name,
                    "position": position,
                    "distance": distance_str,
                    "confidence": round(conf, 2)
                })

        return scene_objects

    def process_live_frame(
        self,
        frame: Optional[np.ndarray],
        need_ocr: bool = False,
        need_objects: bool = True,
        need_depth: bool = True
    ) -> Dict[str, Any]:
        """
        Selectively executes OCR and/or YOLO+MiDaS on the live camera frame based on requested capabilities.
        Returns strictly factual observations.
        """
        if frame is None:
            return {
                "text": "",
                "objects": [],
                "warnings": ["Camera frame unavailable"]
            }

        extracted_text = ""
        if need_ocr:
            extracted_text = self.read_text(frame)

        detected_objects = []
        if need_objects:
            detected_objects = self.get_scene_objects(frame, need_depth=need_depth)

        warnings = []
        for obj in detected_objects:
            if "very close" in obj.get("distance", "").lower():
                warnings.append({
                    "type": "proximity",
                    "object": obj.get("label", "Object"),
                    "message": f"{obj.get('label', 'Obstacle')} is very close in {obj.get('position', 'front')}"
                })

        return {
            "text": extracted_text,
            "ocr_text": extracted_text,
            "objects": detected_objects,
            "detected_objects": detected_objects,
            "warnings": warnings
        }
