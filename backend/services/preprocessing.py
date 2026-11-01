"""Image preprocessing. Not wired into the API yet, see docs/preprocessing.md."""
import cv2
from cv2.typing import MatLike
import numpy as np

from services.detection import min_checkbox_side


class PreprocessingService:
    def cleanup_image(self, img: MatLike) -> MatLike:
        """Remove noise, shadows, and other artifacts from the image."""
        # 1. Make the image gray if it isn't already.
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 2. Binarize: ink = 255, paper = 0. Adaptive threshold copes with shading/uneven scans.
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
        )
        return binary


    def estimate_skew(self, gray: np.ndarray, max_angle: float = 15) -> float:
        """Median angle of long, near-horizontal lines (forms are full of table rules)."""
        scale = min(1.0, 1200 / gray.shape[1])                    # estimate on a smaller copy: much faster
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
        w = gray.shape[1]
        lines = cv2.HoughLinesP(binary, 1, np.pi / 1800, threshold=100, minLineLength=w // 4, maxLineGap=10)
        if lines is None:
            return 0.0
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
        angles = [a for a in angles if abs(a) <= max_angle]
        return float(np.median(angles)) if angles else 0.0

    def deskew(self, gray: np.ndarray, min_angle: float = 0.1) -> tuple[np.ndarray, np.ndarray | None]:
        """Rotate the page straight. Returns the straightened image and the inverse transform
        (None when the page was already straight)."""
        angle = self.estimate_skew(gray)
        if abs(angle) < min_angle:
            return gray, None
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        straight = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        return straight, cv2.invertAffineTransform(M)

    def map_boxes_back(self, boxes: list[tuple], inverse: np.ndarray | None) -> list[tuple]:
        """Map (x1, y1, x2, y2) boxes found on the straightened image back to input pixels."""
        if inverse is None:
            return boxes
        out = []
        for x1, y1, x2, y2 in boxes:
            pts = inverse @ np.array([[x1, x2, x2, x1], [y1, y1, y2, y2], [1, 1, 1, 1]])
            out.append((pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()))
        return out


def get_preprocessing_service() -> PreprocessingService:
    return PreprocessingService()
