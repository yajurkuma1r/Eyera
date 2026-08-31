import cv2
import torch
import pytesseract
from ultralytics import YOLO
import supervision as sv

from app.services.vision.approach_detector import ApproachDetector


class VisionService:
    """
    Combined Vision + OCR service for Eyera.

    Provides:
    1. Object detection using YOLO
    2. Depth estimation using MiDaS
    3. Object tracking using ByteTrack
    4. Obstacle / approaching-object warnings
    5. Structured scene understanding
    6. OCR text extraction using Tesseract

    This service can later be connected to the
    Voice Interaction and LLM/Fusion layers.
    """

    def __init__(
        self,
        yolo_model_path="yolov8n.pt",
        close_threshold=300.0,
        very_close_threshold=500.0,
        center_width_ratio=0.4,
        tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ):
        self.yolo_model_path = yolo_model_path
        self.close_threshold = close_threshold
        self.very_close_threshold = very_close_threshold
        self.center_width_ratio = center_width_ratio

        # ==========================
        # Load YOLO
        # ==========================
        print("[VisionService] Loading YOLO model...")
        self.yolo = YOLO(self.yolo_model_path)

        # ==========================
        # Load MiDaS
        # ==========================
        print("[VisionService] Loading MiDaS model...")
        self.midas = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small"
        )
        self.midas.eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms"
        )
        self.transform = transforms.small_transform

        # ==========================
        # Load ByteTrack
        # ==========================
        print("[VisionService] Loading ByteTrack...")
        self.tracker = sv.ByteTrack()

        # ==========================
        # Approach Detector
        # ==========================
        self.approach_detector = ApproachDetector()

        # ==========================
        # Bounding Box Annotator
        # ==========================
        self.box_annotator = sv.BoxAnnotator()

        # ==========================
        # Tesseract OCR
        # ==========================
        print("[VisionService] Setting up Tesseract OCR...")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Friendly names for detected objects
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

    # ============================================================
    # DEPTH ESTIMATION
    # ============================================================

    def _get_depth_map(self, frame):
        """
        Generate a depth map for the given camera frame using MiDaS.
        """

        img_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

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

    def _get_position(self, center_x, frame_width):
        """
        Determine whether an object is on the left,
        center, or right side of the camera view.
        """

        if center_x < frame_width / 3:
            return "left"

        elif center_x < (frame_width / 3) * 2:
            return "center"

        return "right"

    def _get_depth_label(self, depth_value):
        """
        Convert raw MiDaS depth value into a simple
        human-readable depth category.
        """

        if depth_value > 800:
            return "near"

        elif depth_value > 400:
            return "medium"

        return "far"

    # ============================================================
    # EXISTING SAFETY / APPROACH DETECTION
    # ============================================================

    def process_frame(self, frame):
        """
        Processes one camera frame.

        Performs:
        - MiDaS depth estimation
        - YOLO object detection
        - ByteTrack tracking
        - Approach detection
        - Obstacle warning generation

        Returns:
            annotated_frame
            warnings
        """

        # ==========================
        # MiDaS Depth Estimation
        # ==========================

        img_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

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

        yolo_result = self.yolo(
            frame,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(
            yolo_result
        )

        # ==========================
        # ByteTrack Tracking
        # ==========================

        detections = self.tracker.update_with_detections(
            detections
        )

        warnings = []

        if detections.tracker_id is not None:

            height, width = depth_map.shape[:2]

            # Walking-path center region
            left_bound = width * (
                0.5 - self.center_width_ratio / 2
            )

            right_bound = width * (
                0.5 + self.center_width_ratio / 2
            )

            for i in range(len(detections.xyxy)):

                x1, y1, x2, y2 = detections.xyxy[i]

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Safety clamp
                center_x = max(
                    0,
                    min(center_x, width - 1)
                )

                center_y = max(
                    0,
                    min(center_y, height - 1)
                )

                depth_value = float(
                    depth_map[center_y, center_x]
                )

                track_id = int(
                    detections.tracker_id[i]
                )

                class_id = int(
                    detections.class_id[i]
                )

                object_name = self.yolo.names[class_id]

                friendly_name = self.friendly_names.get(
                    object_name,
                    "Obstacle"
                )

                # Track whether object is approaching
                status = self.approach_detector.update(
                    track_id=track_id,
                    depth=depth_value
                )

                is_ahead = (
                    left_bound
                    <= center_x
                    <= right_bound
                )

                # --------------------------
                # Approaching object
                # --------------------------

                if status == "CONFIRMED_APPROACHING":

                    warnings.append({
                        "type": "approaching",
                        "object": friendly_name,
                        "depth": depth_value,
                        "message": f"{friendly_name} approaching"
                    })

                # --------------------------
                # Very close object
                # --------------------------

                elif depth_value > self.very_close_threshold:

                    warnings.append({
                        "type": "very_close",
                        "object": friendly_name,
                        "depth": depth_value,
                        "message": "Object very close"
                    })

                # --------------------------
                # Object directly ahead
                # --------------------------

                elif (
                    is_ahead
                    and depth_value > self.close_threshold
                ):

                    warnings.append({
                        "type": "ahead",
                        "object": friendly_name,
                        "depth": depth_value,
                        "message": f"{friendly_name} ahead"
                    })

        # ==========================
        # Annotate Frame
        # ==========================

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        return annotated_frame, warnings

    # ============================================================
    # STRUCTURED SCENE UNDERSTANDING
    # ============================================================

    def get_scene_objects(self, frame):
        """
        Detect and track objects in a frame.

        Returns a list such as:

        [
            {
                "id": 1,
                "name": "person",
                "position": "center",
                "depth": "near"
            }
        ]
        """

        frame_height, frame_width = frame.shape[:2]

        depth_map = self._get_depth_map(frame)

        result = self.yolo(
            frame,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(
            result
        )

        detections = self.tracker.update_with_detections(
            detections
        )

        scene_objects = []

        if detections.tracker_id is not None:

            for i in range(len(detections.xyxy)):

                x1, y1, x2, y2 = detections.xyxy[i]

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                center_x = max(
                    0,
                    min(center_x, depth_map.shape[1] - 1)
                )

                center_y = max(
                    0,
                    min(center_y, depth_map.shape[0] - 1)
                )

                depth_value = float(
                    depth_map[center_y, center_x]
                )

                track_id = int(
                    detections.tracker_id[i]
                )

                class_id = int(
                    detections.class_id[i]
                )

                object_name = self.yolo.names[class_id]

                scene_objects.append({
                    "id": track_id,
                    "name": object_name,
                    "position": self._get_position(
                        center_x,
                        frame_width
                    ),
                    "depth": self._get_depth_label(
                        depth_value
                    )
                })

        return scene_objects

    # ============================================================
    # OCR / TEXT READING
    # ============================================================

    def read_text(self, frame):
        """
        Extract readable text from a camera frame using Tesseract.

        Returns:
            cleaned text string
            or an empty string if no meaningful text is found.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        raw_text = pytesseract.image_to_string(
            gray
        )

        # Remove empty lines
        lines = [
            line.strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]

        cleaned_text = " ".join(lines)

        # Reject very short OCR results
        if len(cleaned_text) < 4:
            return ""

        # Reject mostly-symbol garbage
        letter_count = sum(
            c.isalpha()
            for c in cleaned_text
        )

        if letter_count / len(cleaned_text) < 0.5:
            return ""

        return cleaned_text

    # ============================================================
    # COMBINED VISION OUTPUT
    # ============================================================

    def get_full_scene(self, frame, include_text=False):
        """
        Return structured visual information for the frame.

        OCR is optional because it is relatively expensive.

        Example output:

        {
            "objects": [
                {
                    "id": 1,
                    "name": "person",
                    "position": "center",
                    "depth": "near"
                }
            ],
            "text": "EXIT"
        }
        """

        return {
            "objects": self.get_scene_objects(frame),
            "text": self.read_text(frame)
            if include_text
            else None
        }