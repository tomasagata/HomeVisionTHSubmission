# Checkbox detection

An HTTP service that finds every checkbox in a document image and says whether it is checked. It comes with a small web frontend that draws the results on top of the page.

```
POST /detect   (multipart/form-data, field "image")

{
  "boxes": [
    {"bbox": [259, 98, 277, 113], "is_checked": false},
    {"bbox": [615, 60, 632, 76],  "is_checked": true}
  ]
}
```

`bbox` is `[x1, y1, x2, y2]`, the top-left and bottom-right corners in pixels of the uploaded image. A file that can't be decoded as an image gets a `400`.

[NOTES.md](NOTES.md) is the chronological log of how I got here: what I tried, what didn't work and why I changed course. This README is the summary.

## Running it

### Try it online!

A hosted version is available
 * frontend: http://54.242.163.45:4173/ # <- this one has a cool UI
 * backend: http://54.242.163.45:8000/

You can try out online by cloning this repo and then doing a curl with

```sh
curl -F image=@backend/tests/data/fullres/images/canvas1.png http://54.242.163.45:4173/detect
```


### With Docker (everything)

```bash
docker compose up --build
```

- Frontend: http://localhost:4173. Drop a page image on it to see the detected boxes (green = checked, blue = unchecked).
- API: the backend isn't published on the host. The frontend proxies `/api/*` to it, so the endpoint can be reached with:

  ```bash
  curl -F image=@backend/tests/data/fullres/images/canvas1.png http://localhost:4173/api/detect
  ```

The trained weights are committed in `backend/models/`, so nothing gets downloaded or trained at build time.

### Locally

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 itself) and Node 22.

```bash
# backend, on http://localhost:8000 (interactive docs at /docs)
cd backend
uv sync
uv run fastapi run main.py
curl -F image=@tests/data/fullres/images/canvas1.png http://localhost:8000/detect

# frontend, on http://localhost:5173, proxies /api to localhost:8000
cd frontend
npm ci
npm run dev

# accuracy report
cd backend
uv run pytest
```

Everything runs on CPU. PyTorch is pulled from the CPU-only wheel index.

## Approach

I split the problem into two stages that can be built, swapped and measured independently:

```
page image ──► detection ──► [x, y, w, h] boxes ──► classification ──► checked / unchecked
               (YOLO)                                (small CNN on each crop)
```

**1. Detection: a fine-tuned YOLO model** (`backend/services/yolo_detection.py`). I fine-tuned Ultralytics' YOLO26s on a public Roboflow dataset of about 2,400 scanned document pages with labelled checkboxes ([cogniquest checkboxes](https://universe.roboflow.com/cogniquest-idc1w/checkboxes-xtv3p/dataset/5)). The dataset has separate classes for checked and unchecked boxes. I merged them into a single "checkbox" class, so the detector only has to answer "where are the boxes?". Training is in `backend/notebooks/yolo.ipynb`.

The detail that mattered most was resizing. The model runs at 1280 px, and the full-resolution pages are up to 2550 × 4200. Ultralytics shrinks large inputs with linear interpolation, which aliases away borders that are 2–3 px thick. Shrinking the page myself with `INTER_AREA` before prediction fixed it: on the densest full-resolution page, recall went from 0.75 to 0.98.

I started with a classic OpenCV detector: adaptive threshold, contours, and keeping square, four-cornered, checkbox-sized shapes. It's still in `backend/services/detection.py`, and the test report scores it next to YOLO. It handles empty boxes well, but a cross or tick drawn through a box breaks its contour. That makes checked boxes the ones it misses, and those are the ones that matter most.

**2. Classification: my own small CNN** (`backend/services/classifier.py`). Each detected box is cropped with 15% padding, converted to grayscale and resized to 64 × 64. A four-layer CNN with about 100k parameters then outputs one logit for "checked". I trained it on [checkbox-cropped-binary](https://huggingface.co/datasets/meghnagera15/checkbox-cropped-binary), with augmentation (padding, slight rotation, contrast, blur, low resolution) to make that dataset look more like document scans. Training is in `backend/notebooks/cnn.ipynb`. The model class and preprocessing live in the service, and the notebook imports them, so training and serving can't drift apart.

I tried a pretrained Hugging Face checkbox classifier first. It was trained on web UI screenshots, barely beat a baseline that answers "unchecked" every time, and was slow. A purpose-built model this small was both faster and much more accurate.

**What counts as checked.** Forms ask for crosses, ticks or filled boxes, and some use a filled box to mean "I changed my mind". With no form-specific rules to go on, I treat a box as checked if it is crossed, ticked or filled.

## Results

**On each model's held-out test split.** Each training notebook ends by scoring its saved model once on its dataset's test split. Neither training nor my tuning saw these pages or crops.

| Model | Test set | Precision | Recall | F1 | Other |
|---|---|---|---|---|---|
| YOLO detector (`yolo.ipynb`) | 204 pages, 2,308 boxes | 0.930 | 0.927 | 0.928 | mAP50 0.947, mAP50-95 0.590 |
| CNN classifier (`cnn.ipynb`), "checked" as positive | 400 crops | 0.984 | 0.910 | 0.945 | accuracy 0.948 |

Ultralytics reports detector precision and recall at the confidence that maximizes F1, not at the 0.25 the API uses. The classifier's errors lean one way: 18 checked crops were called unchecked, against 3 the other way round.

**On the challenge pages.** Measured on the four pages from the challenge PDF, which I labelled in Label Studio (288 checkboxes per set). `fullres` has the pages at original resolution and `lowres` at roughly a third of that. A detection counts as correct at IoU ≥ 0.5.

| | `fullres` | `lowres` |
|---|---|---|
| Detection, YOLO (precision / recall / F1) | 0.98 / 0.98 / 0.98 | 0.98 / 0.99 / 0.98 |
| Detection, OpenCV baseline (precision / recall / F1) | 0.72 / 0.99 / 0.83 | 0.89 / 0.85 / 0.87 |
| Classification on the labelled boxes (accuracy / recall for checked) | 0.99 / 1.00 | 1.00 / 1.00 |
| End to end through `/detect`: correct / wrong / missed / false positives | 280 / 2 / 5 / 5 | 283 / 0 / 4 / 7 |

The first three rows come from `uv run pytest`. I score detection and classification separately on purpose. The classifier is fed the labelled boxes rather than the detector's output, so a detection miss can't show up as a classification error, and I can tell which stage needs work. The last row comes from `backend/notebooks/test.ipynb`, which sends every page through the real route and draws hits, misses and false positives on top of it.

On my desktop CPU (AMD Ryzen 7 5700), a request takes about 0.2 s for a low-resolution page and 0.3 s for a full-resolution one.

## Tradeoffs

- **Learned detector over heuristics.** The OpenCV detector is tiny, explainable and needs no training data, but it wasn't going to handle crossed boxes. YOLO adds about 20 MB of weights and a heavy dependency (PyTorch + Ultralytics, so the Docker image is large). I also can't tune it by reading code. In exchange, recall on checked boxes stopped being a problem. Training took about 10 hours on my GPU, which was a real risk given my time budget. I took it because the heuristics had hit a ceiling.
- **Two stages instead of one detector with checked/unchecked classes.** A single YOLO model with two classes would be simpler to serve. Keeping them apart lets me evaluate each stage on its own, retrain the cheap classifier without touching the detector, and keep my definition of "checked" in one place. It also means I don't depend on how the detection dataset happened to label checked boxes. The cost is a second model and a second pass over the crops, which is negligible next to YOLO.
- **CPU-only inference.** I wanted this to run on an ordinary VM without a GPU. A small YOLO variant at 1280 px is the compromise: the nano model missed small boxes, and the medium one was too slow to train on my hardware.
- **Report-only tests.** The test suite prints precision, recall and accuracy per image and per dataset instead of asserting thresholds. With 4 pages, a hard threshold would be either meaningless or flaky. Adding a labelled dataset folder under `backend/tests/data/` adds it to the report with no code changes.
- **Frontend proxy instead of CORS.** The frontend calls `/api` on its own origin and Vite proxies it to the backend, so the API doesn't need CORS configuration.

## Known limitations

- **The evaluation set is tiny and not independent.** It's four pages, and I used them to make decisions such as image size, confidence threshold and the resize fix. The challenge-page numbers are optimistic, and I'd expect them to drop on unseen forms. The test-split scores (F1 0.928 for detection, 0.945 for classification) are the more honest estimate, though those datasets aren't the target forms either.
- **The labels are mine.** Some boxes are genuinely ambiguous. Those count for detection but are excluded from classification.
- **Images only.** PDFs and multi-page documents aren't accepted. The client has to rasterize pages first.
- **No deskewing or rotation handling.** `backend/services/preprocessing.py` has a deskew step, but it isn't wired into the API, and I haven't tested the pipeline on skewed or rotated scans.
- **Fixed thresholds and no confidence in the response.** The detector uses a fixed confidence of 0.25, and the classifier uses a logit threshold of 0. The response only has a boolean, so a client can't separate "clearly checked" from "barely".
- **Very small boxes on very large pages.** Every page is shrunk to 1280 px on its long side. On a large page, a box that's only a few pixels wide can disappear.
- **Not hardened for production.**
  - There's no upload size limit, authentication or rate limiting, and the whole upload is read into memory.
  - Inference runs inside an `async` handler, so a slow request blocks the event loop for every other request.

## What I'd do next

1. Label a larger, independent set of real appraisal pages (skewed, photographed, faxed), and measure on that before tuning anything else.
2. Move inference off the event loop: a sync handler or a worker pool, plus a request size limit.
3. Return a confidence per box, and accept PDFs by rasterizing each page server-side.
4. Wire deskewing in, and retrain the detector with rotation and scan-noise augmentation.
5. Export the models to ONNX to drop PyTorch from the runtime image.

## Layout

```
backend/     FastAPI service, models, training notebooks, accuracy tests (see backend/README.md)
frontend/    React + Vite page that uploads an image and draws the returned boxes
docker-compose.yml
NOTES.md     how the project evolved, step by step
```
