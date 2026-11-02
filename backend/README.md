# Checkbox detection: backend

A FastAPI service that takes a document image and returns every checkbox it finds, along with whether each one is checked. The approach, results and limitations are in the [top-level README](../README.md). This file covers how the backend is organised and tested.

```
POST /detect   (multipart, field "image")
-> {"boxes": [{"bbox": [x1, y1, x2, y2], "is_checked": true}, ...]}
```

## Running

```bash
uv sync
uv run fastapi run main.py    # API on http://localhost:8000
uv run pytest                 # accuracy report
```

Both models are loaded at startup, in `main.py`'s lifespan, so the first request is as fast as the rest.

## Layout

```
main.py                   FastAPI app, loads both models at startup
routes/detection.py       POST /detect: decode upload -> detect -> classify
schemas/checkbox.py       response models
services/
  yolo_detection.py       checkbox detection with the fine-tuned YOLO model (used by the API)
  classifier.py           checked / unchecked CNN: model, preprocessing, inference (used by the API)
  detection.py            OpenCV contour detector, kept as a baseline and scored next to YOLO
  preprocessing.py        binarization for the OpenCV detector, plus deskewing (not wired in yet)
models/
  checkbox_yolo.pt        detector weights, trained in notebooks/yolo.ipynb
  checkbox_cnn.pt         classifier weights, trained in notebooks/cnn.ipynb
notebooks/
  yolo.ipynb              detector training and evaluation
  cnn.ipynb               classifier training and evaluation on the dataset's test split
  test.ipynb              end-to-end check: every test page goes through /detect, results drawn on the page
tests/                    accuracy tests for detection and classification
  data/                   labelled test images, one folder per dataset (see tests/data/README.md)
```

Internally, boxes travel as COCO `[x, y, w, h]`, the same format as the labels. They're converted to the API's corner format `[x1, y1, x2, y2]` when the classifier builds the response objects.

## Testing

The tests are built around labelled data rather than hand-written assertions. Every folder under `tests/data/` with an `annotations.json` is picked up automatically, and each image becomes its own test case. That way adding a new dataset (skewed scans, low-res photos, …) needs no test code changes.

The suites are report-only on purpose. A test fails only if the data is broken or the pipeline crashes. The actual numbers are printed in a summary table at the end of the run, per image and per dataset:

- **Detection**: predictions are matched one-to-one against labelled boxes at IoU ≥ 0.5. I report precision, recall and F1 for both the YOLO detector and the OpenCV baseline. Ambiguous checkboxes count as checkboxes here.
- **Classification**: the classifier is fed the *labelled* boxes, not the detector's output, so a detection miss doesn't show up as a classification error. I report accuracy, plus precision and recall for the "checked" class, because the labels are skewed (~68% unchecked) and accuracy alone flatters a classifier that always says "unchecked". Ambiguous boxes are excluded because there is no right answer for them.

`notebooks/test.ipynb` complements this by running the full pipeline through the route, the way a client would, and drawing every hit, miss and false positive on the page.

The four challenge pages are too few to judge a model on, and I used them to make decisions. So `notebooks/yolo.ipynb` and `notebooks/cnn.ipynb` each end by scoring their saved model once on its dataset's held-out test split: F1 0.928 for the detector and 0.945 for the classifier. The root README has the full numbers.
