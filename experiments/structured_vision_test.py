import cv2
import torch
from ultralytics import YOLO
import supervision as sv

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
# Helper: turn bounding box x-position into left/center/right
# ==========================
def get_position(center_x, frame_width):
    if center_x < frame_width / 3:
        return "left"
    elif center_x < (frame_width / 3) * 2:
        return "center"
    else:
        return "right"

# ==========================
# Helper: turn raw depth number into near/medium/far
# ==========================
def get_depth_label(depth_value):
    if depth_value > 800:
        return "near"
    elif depth_value > 400:
        return "medium"
    else:
        return "far"

# ==========================
# Webcam
# ==========================
cap = cv2.VideoCapture(0)
box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    # MiDaS Depth Estimation
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

    # YOLO Detection
    result = yolo(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # ByteTrack Tracking
    detections = tracker.update_with_detections(detections)

    # ==========================
    # Build structured output for this frame
    # ==========================
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
            object_name = yolo.names[class_id]

            scene_objects.append({
                "id": int(track_id),
                "name": object_name,
                "position": get_position(center_x, frame_width),
                "depth": get_depth_label(depth_value)
            })

    # Print the structured data for this frame
    print(scene_objects)

    # Draw Bounding Boxes
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )
    cv2.imshow("EYERA Structured Vision Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()