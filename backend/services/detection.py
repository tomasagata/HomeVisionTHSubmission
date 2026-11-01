import cv2
from cv2.typing import MatLike
import numpy as np

CONTOUR_AREA_THRESHOLD = 0.85  # square corners, not a round letter hole
ASPECT_RATIO_THRESHOLD = 16/10 # more or less square, not a long rectangle

def find_candidates(binary: MatLike) -> list[tuple[int, int, int, int]]:
    h_img, w_img = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not checkbox_sized(h_img, w_img, w, h):
            continue
        if not square_shaped(w, h):
            continue
        if fill_ratio(c, w, h) < CONTOUR_AREA_THRESHOLD:
            continue
        boxes.append((x, y, x + w, y + h))
    return dedupe(boxes)

def fill_ratio(c, w, h):
    # A regular checkbox is a square or rectangle.
    # So the area inside the contour should be close to the
    # area of a regular perfect rectangle of the same width and height.
    # Here it's divided by (w-1)*(h-1) because the contour is 1 pixel wide,
    # so the area of an equivalent rectangle is (w-1)*(h-1).
    return cv2.contourArea(c) / ((w-1) * (h-1))

def square_shaped(w, h):
    return 1/ASPECT_RATIO_THRESHOLD <= w / h <= ASPECT_RATIO_THRESHOLD
    
def min_checkbox_side(w_img):
    # tbh this is a bit of a hack, but for now it works.
    return max(8, int(0.006 * w_img))   # size limits relative to page width

def checkbox_sized(h_img, w_img, w, h):
    min_side = min_checkbox_side(w_img)
    max_side = int(0.03 * w_img)
    return min_side <= w <= max_side and min_side <= h <= max_side

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def dedupe(boxes, thr=0.5):
    # A box border has an outer AND inner contour -> keep the larger one of overlapping pairs.
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in boxes:
        if all(iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


class DetectionService:
    def find_checkboxes(self, binary: MatLike) -> list[list[int]]:
        """Return checkbox bounding boxes as [x, y, w, h] (same convention as COCO and the API)."""
        return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in find_candidates(binary)]


def get_detection_service() -> DetectionService:
    return DetectionService()
