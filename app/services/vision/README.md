# Vision Service — Visual Understanding & OCR

This module gives structured information about what the camera sees:
detected objects (with position/depth) and any readable text.

## How to use it

```python
from app.services.vision.vision_service import VisionService

vision = VisionService()  # create once - loads YOLO/MiDaS/Tesseract

# For a single camera frame:
scene = vision.get_full_scene(frame, include_text=True)
```

**Important:** create `VisionService()` only once (loading the models is slow).
Then call its methods repeatedly on new frames.

## Output format

`get_full_scene(frame, include_text=True)` returns:

```python
{
    "objects": [
        {
            "id": 1,              # tracking ID (same object = same ID across frames)
            "name": "person",     # object class name
            "position": "center", # "left" / "center" / "right"
            "depth": "near"       # "near" / "medium" / "far"
        },
        ...
    ],
    "text": "EXIT"  # any text read from the frame, or "" if none/unclear
}
```

## Other available methods

- `get_scene_objects(frame)` → just the objects list (fast, no OCR)
- `read_text(frame)` → just the text string (slower — only call when needed)

## Notes

- `include_text=False` (default) skips OCR entirely — use this for continuous
  scanning, and only set it to `True` when the user actually asks to read
  something (e.g. a "read the menu" command), since OCR is slow.
- Text output is filtered to remove OCR noise/garbage from busy backgrounds —
  short or mostly-symbol results return `""` instead of junk text.
- FastSAM/segmentation is not yet integrated — parked for a future milestone.