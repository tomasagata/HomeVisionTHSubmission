from pydantic import BaseModel

class Checkbox(BaseModel):
    bbox: list[int]
    is_checked: bool