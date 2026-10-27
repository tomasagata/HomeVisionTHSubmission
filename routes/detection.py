from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile

from services.detection import DetectionService, get_detection_service

router = APIRouter(
    prefix="/",
)

@router.post("/")
async def detect_checkboxes(
    image: UploadFile, 
    detectionService: Annotated[DetectionService, Depends(get_detection_service)]
):
    detected_checkboxes = detectionService.detect_checkboxes(image)
    return detected_checkboxes
