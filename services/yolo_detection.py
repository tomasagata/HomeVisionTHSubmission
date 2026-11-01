"""Checkbox detection with the YOLO model trained in yolo.ipynb."""
from functools import cache
from pathlib import Path

import cv2
from cv2.typing import MatLike
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "checkbox_yolo.pt"
IMGSZ = 1280  # checkboxes are only a few dozen pixels, downscaling the page would lose them
CONF = 0.25


class DetectionService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = YOLO(model_path)

    def find_checkboxes(self, gray: MatLike) -> list[list[int]]:
        """Return checkbox bounding boxes as [x, y, w, h] (same convention as COCO and the API).

        Takes the grayscale page, not the binarized one: the model was trained on raw scans."""
        result = self.model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        return [[round(x1), round(y1), round(x2 - x1), round(y2 - y1)] for x1, y1, x2, y2 in result.boxes.xyxy.tolist()]


@cache  # load the weights once per process, not on every request
def get_detection_service() -> DetectionService:
    return DetectionService()
