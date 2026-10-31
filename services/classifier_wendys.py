from functools import cache

import numpy as np
from PIL import Image
from transformers import pipeline

from schemas.checkbox import Checkbox

MODEL_ID = "wendys-llc/checkbox-classifier"  # the model card's "-efficientnet" id doesn't exist
CROP_PADDING = 0.15  # fraction of the box added on each side, so the whole border is in the crop


def crop(gray: np.ndarray, bbox: list[int]) -> Image.Image:
    x, y, w, h = bbox
    pad_x, pad_y = round(w * CROP_PADDING), round(h * CROP_PADDING)
    h_img, w_img = gray.shape
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
    # The model was trained on RGB screenshots.
    return Image.fromarray(gray[y1:y2, x1:x2]).convert("RGB")


class WendysCheckboxClassifierService:
    def __init__(self):
        self.classifier = pipeline("image-classification", model=MODEL_ID)

    def classify_checkboxes(self, gray: np.ndarray, bboxes: list[list[int]]) -> list[Checkbox]:
        if not bboxes:
            return []
        predictions = self.classifier([crop(gray, bbox) for bbox in bboxes], top_k=1)
        return [
            Checkbox(bbox=list(bbox), is_checked=prediction[0]["label"] == "checked")
            for bbox, prediction in zip(bboxes, predictions, strict=True)
        ]


@cache  # load the model once per process, not on every request
def get_classifier_service() -> WendysCheckboxClassifierService:
    return WendysCheckboxClassifierService()
