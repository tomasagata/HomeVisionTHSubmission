"""Checkbox detection with the YOLO model trained in yolo.ipynb."""
from functools import cache
from pathlib import Path

import cv2
from cv2.typing import MatLike
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "checkbox_yolo.pt"
IMGSZ = 1280  # checkboxes are only a few dozen pixels, downscaling the page would lose them
CONF = 0.39  # maximizes F1 on the training dataset's validation split


class DetectionService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = YOLO(model_path)

    def find_checkboxes(self, gray: MatLike) -> list[list[int]]:
        """Return checkbox bounding boxes as [x, y, w, h] (same convention as COCO and the API).

        Takes the grayscale page, not the binarized one: the model was trained on raw scans.

        Pages larger than IMGSZ are shrunk here with INTER_AREA. Ultralytics would shrink them with INTER_LINEAR,
        which aliases on full-resolution scans (~3x) and breaks the thin checkbox borders."""
        height, width = gray.shape
        scale = min(1.0, IMGSZ / max(height, width))
        if scale < 1:
            # max(1, ...): on very thin pages (e.g. 4000x1) the short side would otherwise round to 0, which cv2 rejects
            gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        result = self.model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        return [[round(x1 / scale), round(y1 / scale), round((x2 - x1) / scale), round((y2 - y1) / scale)]
                for x1, y1, x2, y2 in result.boxes.xyxy.tolist()]


@cache  # load the weights once per process, not on every request
def get_detection_service() -> DetectionService:
    return DetectionService()
