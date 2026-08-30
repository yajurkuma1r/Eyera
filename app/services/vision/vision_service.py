import cv2
import numpy as np
import pytesseract
import torch
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
        yolo_model_path="yolov8n.pt",
        close_threshold=300.0,
        very_close_threshold=500.0,
        center_width_ratio=0.4,
        tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ):
        # Support callers passing tesseract_path positionally as first argument
        if isinstance(yolo_model_path, str) and ("tesseract" in yolo_model_path.lower() or yolo_model_path.endswith(".exe")):
            tesseract_path = yolo_model_path
            yolo_model_path = "yolov8n.pt"

        self.yolo_model_path = yolo_model_path
        self.close_threshold = close_threshold
        self.very_close_threshold = very_close_threshold
        self.center_width_ratio = center_width_ratio
        self.tesseract_path = tesseract_path

        # Set up Tesseract
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Load YOLO model
        print("[VisionService] Loading YOLO model...")
        self.yolo = YOLO(self.yolo_model_path)

        # Load MiDaS model
        print("[VisionService] Loading MiDaS model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        # Initialize ByteTrack
        print("[VisionService] Loading ByteTrack...")
        self.tracker = sv.ByteTrack()

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

    def _get_position(self, center_x, frame_width):
        if center_x < frame_width / 3:
            return "left"
        elif center_x < (frame_width / 3) * 2:
            return "center"
        else:
            return "right"

    def _get_depth_label(self, depth_value):
        if depth_value > 800:
            return "near"
        elif depth_value > 400:
            return "medium"
        else:
            return "far"

    def _get_depth_map(self, frame):
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

    def get_scene_objects(self, frame):
        """
        Takes a camera frame, returns a list of structured object dicts:
        [{"id": int, "name": str, "position": str, "depth": str}, ...]
        """
        frame_height, frame_width = frame.shape[:2]
        depth_map = self._get_depth_map(frame)

        result = self.yolo(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)

        scene_objects = []
        if detections.tracker_id is not None:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center_x_clamped = max(0, min(center_x, depth_map.shape[1] - 1))
                center_y_clamped = max(0, min(center_y, depth_map.shape[0] - 1))
                depth_value = depth_map[center_y_clamped, center_x_clamped]

                track_id = detections.tracker_id[i]
                class_id = int(detections.class_id[i])
                object_name = self.yolo.names[class_id]

                scene_objects.append({
                    "id": int(track_id),
                    "name": object_name,
                    "position": self._get_position(center_x, frame_width),
                    "depth": self._get_depth_label(depth_value)
                })
        return scene_objects

    def read_text(self, frame):
        """
        Takes a camera frame, returns any text detected in it as a string.
        Filters out OCR noise/garbage (short fragments, mostly symbols)
        that comes from Tesseract misreading textures or busy backgrounds.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_text = pytesseract.image_to_string(gray)

        # Clean up: remove empty lines, join into one string
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        cleaned_text = " ".join(lines)

        # Reject obvious noise:
        # - too short to be meaningful
        # - mostly non-letter characters (symbols/garbage)
        if len(cleaned_text) < 4:
            return ""

        letter_count = sum(c.isalpha() for c in cleaned_text)
        if letter_count / len(cleaned_text) < 0.5:
            return ""

        return cleaned_text

    def get_full_scene(self, frame, include_text=False):
        """
        Returns the combined structured output for one frame.
        Set include_text=True only when text-reading is actually needed
        (e.g. triggered by a READ_MENU command) - OCR is slow, so don't
        run it on every frame.
        """
        return {
            "objects": self.get_scene_objects(frame),
            "text": self.read_text(frame) if include_text else None
        }

    def process_frame(self, frame):
        """
        Processes a single camera frame.
        Estimates depth, detects objects, tracks them, and evaluates warning conditions.
        Returns:
            annotated_frame: Frame with bounding boxes drawn.
            warnings: List of warning dicts: [{"type": str, "object": str, "depth": float, "message": str}]
        """
        # ==========================
        # MiDaS Depth Estimation
        # ==========================
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

        depth_map = prediction.cpu().numpy()

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
            height, width = depth_map.shape[:2]
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

                depth_value = float(depth_map[center_y, center_x])
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
