import cv2
import torch
from ultralytics import YOLO

# ----------------------
# Load YOLO
# ----------------------
yolo = YOLO("yolov8n.pt")

# ----------------------
# Load MiDaS
# ----------------------
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# ----------------------
# Webcam
# ----------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------
    # MiDaS Depth
    # ----------------------
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

    # ----------------------
    # YOLO Detection
    # ----------------------
    results = yolo(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            object_name = yolo.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            depth_value = depth_map[center_y, center_x]

            print(
                f"{object_name} | Depth Score: {depth_value:.2f}"
            )

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO + MiDaS", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()