from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from schemas.checkbox import DetectionResponse
from services.classifier import CheckboxClassifierService, get_classifier_service
from services.yolo_detection import DetectionService, get_detection_service

router = APIRouter()

@router.post("/detect")
async def detect_checkboxes(
    image: UploadFile,
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
    classifier_service: Annotated[CheckboxClassifierService, Depends(get_classifier_service)]
) -> DetectionResponse:
    gray = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image")
    found = detection_service.find_checkboxes(gray)
    return DetectionResponse(boxes=classifier_service.classify_checkboxes(gray, found))
