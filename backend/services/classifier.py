from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from schemas.checkbox import Checkbox

INPUT_SIZE = 64  # checkboxes carry little detail, and a small input keeps CPU training fast
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "checkbox_cnn.pt"
CROP_PADDING = 0.15  # fraction of the box added on each side, so the whole border is in the crop


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL crop -> 1 x INPUT_SIZE x INPUT_SIZE tensor in [-1, 1]. Shared by training and inference."""
    gray = img.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    pixels = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255)
    return (pixels * 2 - 1).unsqueeze(0)


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)
    )

def crop(gray: np.ndarray, bbox: list[int]) -> Image.Image:
    x, y, w, h = bbox
    pad_x, pad_y = round(w * CROP_PADDING), round(h * CROP_PADDING)
    h_img, w_img = gray.shape
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
    # The model was trained on RGB screenshots.
    return Image.fromarray(gray[y1:y2, x1:x2]).convert("RGB")


class CheckboxCNN(nn.Module):
    """Small CNN that outputs one logit per crop: > 0 means checked."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128)  # 64px -> 4px
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


class CheckboxClassifierService:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = CheckboxCNN()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def classify_checkboxes(self, gray: np.ndarray, bboxes: list[list[int]]) -> list[Checkbox]:
        """Takes [x, y, w, h] boxes (as the detector returns them), returns them as API corners [x1, y1, x2, y2]."""
        if not bboxes:
            return []
        batch = torch.stack([preprocess(crop(gray, bbox)) for bbox in bboxes])
        with torch.inference_mode():
            checked = self.model(batch) > 0
        return [
            Checkbox(bbox=[x, y, x + w, y + h], is_checked=bool(is_checked))
            for (x, y, w, h), is_checked in zip(bboxes, checked, strict=True)
        ]


@cache  # load the weights once per process, not on every request
def get_classifier_service() -> CheckboxClassifierService:
    return CheckboxClassifierService()
