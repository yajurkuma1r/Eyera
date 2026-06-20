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
# Webcam
# ==========================
cap = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ==========================
    # MiDaS Depth Estimation
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
    # YOLO Detection
    # ==========================
    result = yolo(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)

    # ==========================
    # ByteTrack Tracking
    # ==========================
    detections = tracker.update_with_detections(detections)

    # ==========================
    # Print Object + ID + Depth
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

            depth_value = depth_map[center_y, center_x]

            track_id = detections.tracker_id[i]

            class_id = int(detections.class_id[i])

            object_name = yolo.names[class_id]

            print(
                f"ID: {track_id} | "
                f"{object_name} | "
                f"Depth: {depth_value:.2f}"
            )

    # ==========================
    # Draw Bounding Boxes
    # ==========================
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    cv2.imshow(
        "YOLO + MiDaS + ByteTrack",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()