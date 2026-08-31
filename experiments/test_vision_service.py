import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from app.services.vision.vision_service import VisionService

vision = VisionService()

cap = cv2.VideoCapture(0)
print("Press SPACE to get full scene (with text). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Vision Service Test", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        scene = vision.get_full_scene(frame, include_text=True)
        print(scene)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()