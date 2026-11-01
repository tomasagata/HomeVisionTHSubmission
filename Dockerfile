FROM intel/pytorch:cpu-2.14.0-ubuntu26.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# opencv-python (pulled in by ultralytics) needs libGL and glib at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the runtime deps from pyproject.toml, pinned to the base image's torch
# so pip never reinstalls it. torch lives in the system Python, hence --break-system-packages.
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip freeze | grep -E '^(torch|torchvision)==' > /tmp/constraints.txt \
    && pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt -c /tmp/constraints.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt

COPY . .

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
