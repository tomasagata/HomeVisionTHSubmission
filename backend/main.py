from fastapi import FastAPI
from routes.detection import router as detection_router

app = FastAPI()
app.include_router(detection_router)

