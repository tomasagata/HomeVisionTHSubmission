import cv2
from cv2.typing import MatLike
import numpy as np

CONTOUR_AREA_THRESHOLD = 0.85  # square corners, not a round letter hole

def find_candidates(binary: MatLike) -> list[tuple[int, int, int, int]]:
    h_img, w_img = binary.shape
    # tbh this is a bit of a hack, but for now it works. 
    min_side = max(8, int(0.006 * w_img))   # size limits relative to page width
    max_side = int(0.03 * w_img)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not checkbox_sized(min_side, max_side, w, h):
            continue
        if not square_shaped(w, h):
            continue
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        if len(approx) != 4:                           # four corners
            continue
        if cv2.contourArea(c) / (w * h) < CONTOUR_AREA_THRESHOLD:        # square corners, not a round letter hole
            continue
        boxes.append((x, y, x + w, y + h))
    return dedupe(boxes)

def square_shaped(w, h):
    return 0.8 <= w / h <= 1.25

def checkbox_sized(min_side, max_side, w, h):
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
