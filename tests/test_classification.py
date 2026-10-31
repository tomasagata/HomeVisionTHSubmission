"""Classification accuracy, isolated from detection: the classifier is fed the ground-truth boxes.

Ambiguous checkboxes have no correct answer and are excluded. Report-only: metrics are printed
in the terminal summary, the test fails only on bad data or crashes."""
from services.classifier_wendys import get_classifier_service as get_wendys_classifier_service
from services.classifier_own import get_classifier_service as get_own_classifier_service
from services.preprocessing import get_preprocessing_service
from tests.evaluation import AMBIGUOUS, CHECKED, ClassificationResult


def test_pretrained_classification(sample, report):
    labelled = [a for a in sample.annotations if a.label != AMBIGUOUS]
    assert labelled, f"{sample.id} has no checked/unchecked annotations"
    binary = get_preprocessing_service().cleanup_image(sample.load_gray())

    bboxes = [[round(v) for v in a.bbox] for a in labelled]
    predictions = get_wendys_classifier_service().classify_checkboxes(binary, bboxes)

    result = ClassificationResult()
    for annotation, prediction in zip(labelled, predictions, strict=True):
        result.add(prediction.is_checked, annotation.label == CHECKED)
    report.classification[sample.id] = result

def test_own_classification(sample, report):
    labelled = [a for a in sample.annotations if a.label != AMBIGUOUS]
    assert labelled, f"{sample.id} has no checked/unchecked annotations"
    binary = get_preprocessing_service().cleanup_image(sample.load_gray())

    bboxes = [[round(v) for v in a.bbox] for a in labelled]
    predictions = get_own_classifier_service().classify_checkboxes(binary, bboxes)

    result = ClassificationResult()
    for annotation, prediction in zip(labelled, predictions, strict=True):
        result.add(prediction.is_checked, annotation.label == CHECKED)
    report.own_classification[sample.id] = result
