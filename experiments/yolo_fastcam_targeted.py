import cv2
from ultralytics import YOLO, FastSAM

# Load models
yolo = YOLO("yolov8n.pt")
fastsam = FastSAM("FastSAM-s.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    annotated_frame = frame.copy()

    # YOLO detections
    results = yolo(frame, verbose=False)[0]

    for box in results.boxes:

        class_id = int(box.cls[0])
        object_name = yolo.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Ignore tiny detections
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        # Crop only detected object
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        try:
            # FastSAM only on cropped object
            fs_results = fastsam(
                crop,
                imgsz=256,
                conf=0.4,
                verbose=False
            )

            segmented_crop = fs_results[0].plot()

            # Put segmented crop back
            segmented_crop = cv2.resize(
                segmented_crop,
                (x2 - x1, y2 - y1)
            )

            annotated_frame[y1:y2, x1:x2] = segmented_crop

        except Exception as e:
            print(f"FastSAM error: {e}")

        # Draw YOLO box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            object_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "YOLO + Targeted FastSAM",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()