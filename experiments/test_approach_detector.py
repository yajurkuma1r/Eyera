import cv2
import torch
from ultralytics import YOLO
import supervision as sv

from app.services.vision.approach_detector import ApproachDetector

# ==========================
# YOLO
# ==========================
yolo = YOLO("yolov8n.pt")

# ==========================
# ByteTrack
# ==========================
tracker = sv.ByteTrack()

# ==========================
# MiDaS
# ==========================
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# ==========================
# Approach Detector
# ==========================
approach_detector = ApproachDetector()

# ==========================
# Webcam
# ==========================
cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()

print("Approach Detector Test Started")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ==========================
    # MiDaS Depth
    # ==========================
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # ==========================
    # YOLO
    # ==========================
    result = yolo(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)

    # ==========================
    # ByteTrack
    # ==========================
    detections = tracker.update_with_detections(detections)

    # ==========================
    # Approach Detection
    # ==========================
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

            object_name = yolo.names[class_id]

            status = approach_detector.update(
                track_id=track_id,
                depth=depth_value
            )

            print(
                f"ID: {track_id} | "
                f"{object_name} | "
                f"Depth: {depth_value:.2f} | "
                f"{status}"
            )

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    cv2.imshow(
        "Approach Detector Test",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()