# Checkbox detection

A small FastAPI service that takes a document image and returns every checkbox it finds, along with whether each one is checked.

```
POST /detect   (multipart, field "image")
-> [{"bbox": [x, y, w, h], "is_checked": true}, ...]
```

## Approach

I split the problem into two independent stages:

1. **Detection** (`services/detection.py`). Classic OpenCV, no learned model. I binarize the page with an adaptive threshold so uneven lighting and scan shading don't matter much. Then I take every contour and keep the ones that look like a checkbox: roughly square, four corners, filled-in corners (which rejects round letter holes like "o"), and a size relative to the page width. A box border yields both an outer and an inner contour, so overlapping candidates are deduplicated by IoU.
2. **Classification**. Decides whether a detected box is checked. I have two implementations with the same interface, so I can compare them:
   - `services/classifier_wendys.py`: the pretrained `wendys-llc/checkbox-classifier` from Hugging Face, as a baseline. It was trained on RGB screenshots and transfers poorly to scanned forms: it calls almost everything "unchecked".
   - `services/classifier_own.py`: a small CNN of my own (~100k parameters, 64×64 grayscale input). I trained it in `cnn.ipynb` on the [checkbox-cropped-binary](https://huggingface.co/datasets/meghnagera15/checkbox-cropped-binary) dataset. That dataset is mostly UI widgets and tinted photos, so I train on grayscale and use augmentation (crop padding, slight rotation, contrast, blur, low resolution) to bring it closer to document scans. The model class and its preprocessing live in the service module and the notebook imports them, so training and serving can't drift apart. The weights are committed in `models/checkbox_cnn.pt`, so nothing has to be retrained to run the API or the tests. On the dataset's own test split it reaches 0.95 accuracy. Most of the errors there are UI widgets (stars, filled squares, toggles) that never show up on our forms.

I evaluate the two stages separately. The classifier is scored on the *ground-truth* boxes, not on detector output. That way a detection miss doesn't show up as a classification error, and I can tell which stage needs work.

## Layout

```
main.py                 FastAPI app
routes/detection.py     POST /detect : decode upload -> detect -> classify
schemas/checkbox.py     response model
services/
  detection.py          checkbox detection
  classifier_wendys.py  checked / unchecked: pretrained baseline
  classifier_own.py     checked / unchecked: own CNN (model, preprocessing, inference)
  preprocessing.py      deskewing, not wired to the API yet (see docs/preprocessing.md)
models/                 trained weights for the own CNN
cnn.ipynb               training and test-split evaluation for the own CNN
data/                   labelled test images, one folder per dataset (see data/README.md)
tests/                  accuracy tests for detection and classification
docs/                   design notes
```

## Running

```bash
uv sync
uv run fastapi dev main.py        # API on http://localhost:8000
uv run pytest                     # accuracy report
```

## Testing

The tests are built around labelled data rather than hand-written assertions. Every folder under `data/` with an `annotations.json` is picked up automatically, and each image becomes its own test case. That way adding a new dataset (skewed scans, low-res photos, …) needs no test code changes.

The suites are report-only on purpose. A test fails only if the data is broken or the pipeline crashes. The actual numbers are printed in a summary table at the end of the run, per image and per dataset:

- **Detection**: predictions are matched one-to-one against labelled boxes at IoU ≥ 0.5. I report precision, recall and F1. Ambiguous checkboxes count as checkboxes here.
- **Classification**: accuracy, plus precision and recall for the "checked" class. Ambiguous boxes are excluded because there is no right answer for them. I report the per-class numbers because the label distribution is skewed (~68% unchecked), so accuracy alone flatters a classifier that always says "unchecked". The pretrained baseline is close to that case.

## Current results

| Dataset | Detection precision | Detection recall | Classification accuracy (pretrained) | Classification accuracy (own CNN) |
|---|---|---|---|---|
| `fullres` | 1.00 | 0.92 | 0.70 | 0.99 |
| `lowres`  | 0.99 | 0.38 | 0.68 | 1.00 |

The own CNN's recall for "checked" is 1.00 on both datasets. Its only errors are two false positives on `fullres/canvas1.png`.

Detection on the full-resolution images is in good shape: no false positives, and most misses are on `canvas1.png`. The low-resolution copies of the same pages are where it breaks down. At that size many checkbox borders are only a pixel or two thick, and they fall apart after thresholding. On `canvas4.png` nothing is found at all. Improving that is what I'm focusing on next. Separately, `/detect` still uses the pretrained classifier and hands it the binarized page. To switch the API to the own CNN, the route has to pass the grayscale page instead, because that's the input the CNN was trained on. The preprocessing stage described in `docs/preprocessing.md` is meant to handle these harder inputs.
