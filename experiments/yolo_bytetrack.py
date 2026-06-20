import cv2
from ultralytics import YOLO
import supervision as sv

# Load YOLO
model = YOLO("yolov8n.pt")

# Create ByteTrack tracker
tracker = sv.ByteTrack()

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO inference
    results = model(frame, verbose=False)[0]

    # Convert YOLO results to Supervision detections
    detections = sv.Detections.from_ultralytics(results)

    # Update tracker
    detections = tracker.update_with_detections(detections)

    # Print tracked objects
    for tracker_id, class_id in zip(
        detections.tracker_id,
        detections.class_id
    ):
        object_name = model.names[int(class_id)]

        print(
            f"Track ID: {tracker_id} | Object: {object_name}"
        )

    # Draw boxes
    annotated_frame = frame.copy()

    box_annotator = sv.BoxAnnotator()

    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    cv2.imshow("YOLO + ByteTrack", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()