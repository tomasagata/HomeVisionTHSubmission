import numpy as np

from schemas.checkbox import Checkbox


class ClassifierService:
    def classify_checkboxes(self, gray: np.ndarray, bboxes: list[list[int]]) -> list[Checkbox]:
        return [
            Checkbox(bbox=list(bbox), is_checked=self.is_checkbox_checked(gray, bbox))
            for bbox in bboxes
        ]

    def is_checkbox_checked(self, gray: np.ndarray, bbox: list[int]) -> bool:
        # TODO: implement an actual classifier here
        return False


def get_classifier_service() -> ClassifierService:
    return ClassifierService()
