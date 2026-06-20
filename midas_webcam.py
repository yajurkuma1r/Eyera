import cv2
import torch

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Load transforms
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize for display
    depth_map = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    cv2.imshow("MiDaS Depth Map", depth_map)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()