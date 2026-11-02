from contextlib import asynccontextmanager

from fastapi import FastAPI
from routes.detection import router as detection_router
from services.classifier import get_classifier_service
from services.yolo_detection import get_detection_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The getters are cached, so calling them here loads the weights before the first request is served.
    get_detection_service()
    get_classifier_service()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(detection_router)
