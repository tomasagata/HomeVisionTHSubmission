from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from routes.detection import router as detection_router
from services.classifier import get_classifier_service
from services.yolo_detection import IMGSZ, get_detection_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The getters are cached, so calling them here loads the weights before the first request is served.
    # Loading isn't enough: the first inference pays a one-off setup cost (~3s on CPU), so run each model once.
    # A blank page has no detections, so the classifier gets a box of its own.
    blank = np.full((IMGSZ, IMGSZ), 255, np.uint8)
    get_detection_service().find_checkboxes(blank)
    get_classifier_service().classify_checkboxes(blank, [[0, 0, 32, 32]])
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(detection_router)
