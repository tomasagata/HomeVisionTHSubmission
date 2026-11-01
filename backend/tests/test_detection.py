"""Detection accuracy: predicted boxes vs. every labelled checkbox (ambiguous ones included).
The OpenCV contour detector and the YOLO model are scored side by side.

Report-only: metrics are printed in the terminal summary, the test fails only on bad data or crashes."""
from services.detection import get_detection_service
from services.preprocessing import get_preprocessing_service
from services.yolo_detection import get_detection_service as get_yolo_detection_service
from tests.evaluation import DetectionResult, match


def test_detection(sample, report):
    assert sample.annotations, f"{sample.id} has no annotations"
    gray = sample.load_gray()
    binary = get_preprocessing_service().cleanup_image(gray)

    preds = get_detection_service().find_checkboxes(binary)
    truths = [a.bbox for a in sample.annotations]

    report.detection[sample.id] = DetectionResult(match(preds, truths), len(preds), len(truths))


def test_yolo_detection(sample, report):
    assert sample.annotations, f"{sample.id} has no annotations"
    gray = sample.load_gray()  # YOLO was trained on raw scans, so no binarization

    preds = get_yolo_detection_service().find_checkboxes(gray)
    truths = [a.bbox for a in sample.annotations]

    report.yolo_detection[sample.id] = DetectionResult(match(preds, truths), len(preds), len(truths))
