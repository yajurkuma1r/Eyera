import pytesseract
from PIL import Image

# Point pytesseract to where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path to your test image — CHANGE THIS to your actual screenshot path
image_path = r"C:\Users\parna\OneDrive\Desktop\test.png"

image = Image.open(image_path)
text = pytesseract.image_to_string(image)

print("Detected text:")
print(text)