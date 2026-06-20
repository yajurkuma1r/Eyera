from ultralytics import YOLO
import cv2

print("EYERA TEST SCRIPT RUNNING")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Keep track of objects already announced
announced_objects = set()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Disable YOLO spam logs
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            # Debug
            print("DEBUG:", object_name)

            # Only announce once
            if object_name not in announced_objects:
                print(f"NEW OBJECT: {object_name}")
                announced_objects.add(object_name)

    annotated_frame = results[0].plot()

    cv2.imshow("Eyera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()