import cv2
import pytesseract

# Point pytesseract to where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def read_text_from_frame(frame):
    """
    Takes a single camera frame (as a numpy array from OpenCV)
    and returns any text detected in it as a string.
    """
    # Convert to grayscale - OCR works better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()


# ==========================
# Simple webcam test loop
# ==========================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    print("Press SPACE to read text from the current frame. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("OCR Webcam Test - press SPACE to read, q to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            detected_text = read_text_from_frame(frame)
            print("Detected text:", detected_text if detected_text else "(nothing readable)")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()