import cv2
import torch
import pytesseract
from ultralytics import YOLO
import supervision as sv


class VisionService:
    """
    Vision Understanding & OCR service for Eyera.
    Loads YOLO, MiDaS, ByteTrack, and Tesseract once, then exposes
    simple functions to get structured scene data from camera frames.
    """

    def __init__(self, tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        # Load YOLO
        self.yolo = YOLO("yolov8n.pt")

        # Load ByteTrack
        self.tracker = sv.ByteTrack()

        # Load MiDaS
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        # Set up Tesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

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