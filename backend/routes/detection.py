from http import HTTPStatus
from io import BytesIO
from typing import Annotated

import anyio
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from schemas.checkbox import DetectionResponse
from services.classifier import CheckboxClassifierService, get_classifier_service
from services.yolo_detection import DetectionService, get_detection_service

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
MAX_PIXELS = 50_000_000  # a Letter page scanned at 600 dpi is ~34 MP; also caps how far a compressed upload can inflate
SUPPORTED_FORMATS = ("PNG", "JPEG", "BMP", "TIFF")  # detected by Pillow from the file's contents, not the client's claim
MAX_QUEUED = 4  # requests allowed to wait for the model before new ones get a 503

# The YOLO detector already uses every core, so queue the requests
INFERENCE_LIMITER = anyio.CapacityLimiter(1)


@router.post("/detect")
async def detect_checkboxes(
    image: UploadFile,
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
    classifier_service: Annotated[CheckboxClassifierService, Depends(get_classifier_service)],
) -> DetectionResponse:
    contents = await image.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail="File size exceeds the maximum limit of 10 MB")

    try:
        with Image.open(BytesIO(contents), formats=SUPPORTED_FORMATS) as header:
            width, height = header.size
    except UnidentifiedImageError:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Unsupported file type. Please upload an image with the following formats: {}.".format(", ".join(SUPPORTED_FORMATS)))
    except Image.DecompressionBombError:
        width, height = MAX_PIXELS, MAX_PIXELS
    if width * height > MAX_PIXELS:
        raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail=f"Image exceeds the maximum of {MAX_PIXELS // 1_000_000} megapixels")

    if INFERENCE_LIMITER.statistics().tasks_waiting >= MAX_QUEUED:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Server is busy, please try again shortly")

    def detect() -> DetectionResponse:
        gray = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Could not decode the uploaded file as an image")
        found = detection_service.find_checkboxes(gray)
        return DetectionResponse(boxes=classifier_service.classify_checkboxes(gray, found))

    return await anyio.to_thread.run_sync(detect, limiter=INFERENCE_LIMITER)
