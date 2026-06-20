import cv2
from ultralytics import FastSAM

model = FastSAM("FastSAM-s.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        imgsz=640,
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("FastSAM", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()