FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    OPENCV_IO_MAX_IMAGE_PIXELS=50000000 \
    PATH="/app/.venv/bin:$PATH"

# opencv-python (pulled in by ultralytics) needs libGL and glib at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
