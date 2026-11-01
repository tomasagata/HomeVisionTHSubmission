from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from schemas.checkbox import Checkbox
from services.classifier import CheckboxClassifierService, get_classifier_service
from services.detection import DetectionService, get_detection_service
from services.preprocessing import PreprocessingService, get_preprocessing_service

router = APIRouter()

@router.post("/detect")
async def detect_checkboxes(
    image: UploadFile,
    preprocessing_service: Annotated[PreprocessingService, Depends(get_preprocessing_service)],
    detection_service: Annotated[DetectionService, Depends(get_detection_service)],
    classifier_service: Annotated[CheckboxClassifierService, Depends(get_classifier_service)]
) -> list[Checkbox]:
    gray = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded file as an image")
    binary = preprocessing_service.cleanup_image(gray)
    found = detection_service.find_checkboxes(binary)
    return classifier_service.classify_checkboxes(binary, found)