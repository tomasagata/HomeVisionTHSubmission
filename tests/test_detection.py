"""Detection accuracy: predicted boxes vs. every labelled checkbox (ambiguous ones included).

Report-only: metrics are printed in the terminal summary, the test fails only on bad data or crashes."""
from services.detection import get_detection_service
from tests.evaluation import DetectionResult, match


def test_detection(sample, report):
    assert sample.annotations, f"{sample.id} has no annotations"
    gray = sample.load_gray()

    preds = get_detection_service().find_checkboxes(gray)
    truths = [a.bbox for a in sample.annotations]

    report.detection[sample.id] = DetectionResult(match(preds, truths), len(preds), len(truths))
