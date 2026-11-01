"""Ground-truth loading, box matching and metrics shared by the accuracy tests."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from services.detection import iou

DATA_DIR = Path(__file__).resolve().parent / "data"
IOU_THRESHOLD = 0.5

# Category names as exported by Label Studio.
CHECKED = "checked checkbox"
UNCHECKED = "unchecked checkbox"
AMBIGUOUS = "ambiguous checkbox"


@dataclass
class Annotation:
    bbox: list[float]  # COCO [x, y, w, h]
    label: str


@dataclass
class Sample:
    dataset: str
    path: Path
    annotations: list[Annotation]

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.path.name}"

    def load_gray(self) -> np.ndarray:
        gray = cv2.imread(str(self.path), cv2.IMREAD_GRAYSCALE)
        assert gray is not None, f"could not read {self.path}: check file_name in annotations.json"
        return gray


def discover_datasets(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(p.parent for p in data_dir.glob("*/annotations.json"))


def load_dataset(dataset_dir: Path) -> list[Sample]:
    coco = json.loads((dataset_dir / "annotations.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    samples = {}
    for img in coco["images"]:
        path = dataset_dir / img["file_name"]
        if not path.exists():  # raw Label Studio exports point outside the dataset folder
            path = dataset_dir / "images" / Path(img["file_name"]).name
        samples[img["id"]] = Sample(dataset_dir.name, path, [])
    for a in coco["annotations"]:
        samples[a["image_id"]].annotations.append(Annotation(a["bbox"], cats[a["category_id"]]))
    return list(samples.values())


def load_all_samples(data_dir: Path = DATA_DIR) -> list[Sample]:
    return [s for d in discover_datasets(data_dir) for s in load_dataset(d)]


def xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def match_pairs(preds, truths, thr=IOU_THRESHOLD) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of [x, y, w, h] boxes: each truth can be claimed by at most one prediction.
    Returns (prediction index, truth index) pairs."""
    truths = [xyxy(t) for t in truths]
    used, pairs = set(), []
    for p_i, p in enumerate(map(xyxy, preds)):
        best, best_i = 0.0, None
        for i, t in enumerate(truths):
            if i not in used and (score := iou(p, t)) > best:
                best, best_i = score, i
        if best >= thr:
            used.add(best_i)
            pairs.append((p_i, best_i))
    return pairs


def match(preds, truths, thr=IOU_THRESHOLD) -> int:
    return len(match_pairs(preds, truths, thr))


def ratio(num, den):
    return num / den if den else None


@dataclass
class DetectionResult:
    tp: int = 0
    n_pred: int = 0
    n_gt: int = 0

    def __add__(self, other):
        return DetectionResult(self.tp + other.tp, self.n_pred + other.n_pred, self.n_gt + other.n_gt)

    @property
    def precision(self):
        return ratio(self.tp, self.n_pred)

    @property
    def recall(self):
        return ratio(self.tp, self.n_gt)

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return ratio(2 * p * r, p + r) if p is not None and r is not None else None


@dataclass
class ClassificationResult:
    """Confusion counts with 'checked' as the positive class."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other):
        return ClassificationResult(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    def add(self, predicted: bool, actual: bool):
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def n(self):
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self):
        return ratio(self.tp + self.tn, self.n)

    @property
    def precision(self):
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self):
        return ratio(self.tp, self.tp + self.fn)


@dataclass
class Report:
    detection: dict[str, DetectionResult] = field(default_factory=dict)
    yolo_detection: dict[str, DetectionResult] = field(default_factory=dict)
    classification: dict[str, ClassificationResult] = field(default_factory=dict)
    own_classification: dict[str, ClassificationResult] = field(default_factory=dict)
