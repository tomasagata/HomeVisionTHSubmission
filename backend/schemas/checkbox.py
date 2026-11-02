from pydantic import BaseModel

class Checkbox(BaseModel):
    bbox: list[int]  # [x1, y1, x2, y2]: top-left and bottom-right corners, in pixels of the uploaded image
    is_checked: bool

class DetectionResponse(BaseModel):
    boxes: list[Checkbox]
