import numpy as np

from schemas.checkbox import Checkbox

class OwnCheckboxClassifierService:
    def __init__(self):
        pass

    def classify_checkboxes(self, gray: np.ndarray, bboxes: list[list[int]]) -> list[Checkbox]:
        return [
            Checkbox(bbox=list(bbox), is_checked=False)
            for bbox in bboxes
        ]

def get_classifier_service() -> OwnCheckboxClassifierService:
    return OwnCheckboxClassifierService()