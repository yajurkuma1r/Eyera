import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from app.services.vision.approach_detector import ApproachDetector

class VisionService:
    def __init__(self, yolo_model_path="yolov8n.pt", close_threshold=300.0, very_close_threshold=500.0, center_width_ratio=0.4):
        self.yolo_model_path = yolo_model_path
        self.close_threshold = close_threshold
        self.very_close_threshold = very_close_threshold
        self.center_width_ratio = center_width_ratio
        
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
