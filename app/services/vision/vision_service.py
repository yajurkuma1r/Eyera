import cv2
import numpy as np
import pytesseract
import torch
from typing import Any, Dict, List, Optional
from ultralytics import YOLO
import supervision as sv

from app.services.vision.approach_detector import ApproachDetector


class VisionService:
    """
    Vision Understanding, Navigation & OCR service for Eyera.
    Combines YOLO object detection, MiDaS depth estimation, ByteTrack tracking,
    approach detection, and Tesseract OCR into a single service.
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        close_threshold: float = 300.0,
        very_close_threshold: float = 500.0,
        center_width_ratio: float = 0.4,
        tesseract_path: Optional[str] = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ):
        # Support callers passing tesseract_path positionally as first argument
        if isinstance(yolo_model_path, str) and ("tesseract" in yolo_model_path.lower() or yolo_model_path.endswith(".exe")):
            tesseract_path = yolo_model_path
            yolo_model_path = "yolov8n.pt"

        self.yolo_model_path = yolo_model_path
        self.close_threshold = close_threshold
        self.very_close_threshold = very_close_threshold
        self.center_width_ratio = center_width_ratio
        self.tesseract_path = tesseract_path or r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # Set up Tesseract
        try:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        except Exception:
            pass

        # Load YOLO model
        print("[VisionService] Loading YOLO model...")
        self.yolo = YOLO(self.yolo_model_path)

        # Load ByteTrack
        print("[VisionService] Loading ByteTrack...")
        self.tracker = sv.ByteTrack()

        # Load MiDaS model
        self.midas = None
        self.transform = None
        self._init_midas()

        # Initialize Approach Detector
        self.approach_detector = ApproachDetector()

        # Bounding box annotator
        self.box_annotator = sv.BoxAnnotator()

        # Class name mapping to friendly names
        self.friendly_names = {
            "person": "Person",
            "car": "Car",
            "truck": "Car",
            "bus": "Car",
            "motorcycle": "Car",
            "bicycle": "Bicycle",
            "traffic light": "Pole",
            "fire hydrant": "Pole",
            "stop sign": "Obstacle",
            "parking meter": "Pole",
            "bench": "Obstacle",
            "chair": "Obstacle",
            "couch": "Obstacle",
            "bed": "Obstacle",
            "dining table": "Obstacle",
        }

    def _init_midas(self):
        try:
            print("[VisionService] Loading MiDaS depth model...")
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False)
            self.midas.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
            self.transform = transforms.small_transform
        except Exception as e:
            print(f"[VisionService] MiDaS load notice ({e}). Using bounding box depth fallback.")
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
        Extracts readable text from the live camera frame using OCR.
        Applies preprocessing (grayscale, contrast threshold) and filters out OCR noise.
        """
        if frame is None:
            return ""

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Contrast enhancement
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            raw_text = pytesseract.image_to_string(processed)

            # Clean lines
            lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
            cleaned_text = " ".join(lines)

            # Noise rejection (too short or mostly symbols)
            if len(cleaned_text) < 3:
                # Try raw grayscale fallback
                raw_text2 = pytesseract.image_to_string(gray)
                lines2 = [line.strip() for line in raw_text2.split("\n") if line.strip()]
                cleaned_text = " ".join(lines2)

            if len(cleaned_text) < 3:
                return ""

            letter_count = sum(c.isalnum() for c in cleaned_text)
            if letter_count / max(len(cleaned_text), 1) < 0.4:
                return ""

            return cleaned_text.strip()
        except Exception as e:
            # Fallback if tesseract binary is not configured
            print(f"[VisionService] OCR runtime notice: {e}")
            return ""

    def get_scene_objects(self, frame: np.ndarray, need_depth: bool = True) -> List[Dict[str, Any]]:
        """
        Runs YOLO object detection and spatial depth calculation on live frame.
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

                # Distance/depth calculation
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
                    "depth": distance_str,
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
        Selectively processes a live camera frame based on requested capabilities.
        Returns structured factual visual data.
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

        # Proximity warnings
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

    def get_full_scene(self, frame: np.ndarray, include_text: bool = False) -> Dict[str, Any]:
        """
        Returns the combined structured output for one frame.
        Set include_text=True only when text-reading is actually needed.
        """
        return {
            "objects": self.get_scene_objects(frame),
            "text": self.read_text(frame) if include_text else None
        }

    def process_frame(self, frame):
        """
        Processes a single camera frame for navigation.
        Estimates depth, detects objects, tracks them, and evaluates warning conditions.
        Returns:
            annotated_frame: Frame with bounding boxes drawn.
            warnings: List of warning dicts: [{"type": str, "object": str, "depth": float, "message": str}]
        """
        # ==========================
        # MiDaS Depth Estimation
        # ==========================
        depth_map = self._get_depth_map(frame)

        # ==========================
        # YOLO Detection
        # ==========================
        yolo_result = self.yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(yolo_result)

        # ==========================
        # ByteTrack Tracking
        # ==========================
        detections = self.tracker.update_with_detections(detections)

        warnings = []

        if detections.tracker_id is not None:
            height, width = (depth_map.shape[:2] if depth_map is not None else frame.shape[:2])
            # Define walking path region (horizontal center region of the frame)
            left_bound = width * (0.5 - self.center_width_ratio / 2)
            right_bound = width * (0.5 + self.center_width_ratio / 2)

            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Bounds safety clamp
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))

                depth_value = float(depth_map[center_y, center_x]) if depth_map is not None else 0.0
                track_id = int(detections.tracker_id[i])
                class_id = int(detections.class_id[i])
                object_name = self.yolo.names[class_id]

                friendly_name = self.friendly_names.get(object_name, "Obstacle")

                # Update approach history
                status = self.approach_detector.update(
                    track_id=track_id,
                    depth=depth_value
                )

                is_ahead = left_bound <= center_x <= right_bound

                # Generate warning conditions
                # 1. Approaching (dynamic warning)
                if status == "CONFIRMED_APPROACHING":
                    warnings.append({
                        "type": "approaching",
                        "object": friendly_name,
                        "depth": depth_value,
                        "message": f"{friendly_name} approaching"
                    })
                # 2. Very Close (danger warning)
                elif depth_value > self.very_close_threshold:
                    warnings.append({
                        "type": "very_close",
                        "object": friendly_name,
                        "depth": depth_value,
                        "message": "Object very close"
                    })
                # 3. Directly Ahead in walking path and close
                elif is_ahead and depth_value > self.close_threshold:
                    warnings.append({
                        "type": "ahead",
                        "object": friendly_name,
                        "depth": depth_value,
                        "message": f"{friendly_name} ahead"
                    })

        # Bounding box annotation
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        return annotated_frame, warnings
