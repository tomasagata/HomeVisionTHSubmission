
from cv2.typing import MatLike


def find_candidates(binary: MatLike) -> list[tuple[int, int, int, int]]:
    pass

class DetectionService:
    def find_checkboxes(self, binary: MatLike) -> list[list[int]]:
        """Return checkbox bounding boxes as [x, y, w, h] (same convention as COCO and the API)."""
        return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in find_candidates(binary)]


def get_detection_service() -> DetectionService:
    return DetectionService()
