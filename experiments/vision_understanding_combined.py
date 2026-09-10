import cv2
import torch
import pytesseract
from ultralytics import YOLO
import supervision as sv

# ==========================
# Setup: YOLO, ByteTrack, MiDaS
# ==========================
yolo = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# ==========================
# Setup: Tesseract OCR
# ==========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_position(center_x, frame_width):
    if center_x < frame_width / 3:
        return "left"
    elif center_x < (frame_width / 3) * 2:
        return "center"
    else:
        return "right"


def get_depth_label(depth_value):
    if depth_value > 800:
        return "near"
    elif depth_value > 400:
        return "medium"
    else:
        return "far"


def read_text_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()


def get_scene_objects(frame, depth_map, frame_width):
    """Runs YOLO + ByteTrack and returns structured object data for one frame."""
    result = yolo(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)

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
    return scene_objects


# ==========================
# Main loop
# ==========================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press SPACE to read text. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # MiDaS depth
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

        # Structured object data (always running)
        scene_objects = get_scene_objects(frame, depth_map, frame_width)

        # Combined structured output
        structured_output = {
            "objects": scene_objects,
            "text": None  # only filled in when SPACE is pressed
        }

        cv2.imshow("EYERA Vision Understanding", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            structured_output["text"] = read_text_from_frame(frame)

        if structured_output["objects"] or structured_output["text"]:
            print(structured_output)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()