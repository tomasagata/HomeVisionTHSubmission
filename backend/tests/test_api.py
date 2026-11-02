"""Checks that /detect rejects hostile uploads cheaply instead of decoding them."""
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import MAX_BODY_SIZE, app

SAMPLE = Path(__file__).parent / "data" / "fullres" / "images" / "canvas1.png"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:  # runs the lifespan, so the models are loaded once for the module
        yield client


def png_header(width: int, height: int) -> bytes:
    """A grayscale PNG that declares width x height but carries a corrupt IDAT: valid enough for the header check,
    and tiny on disk however large it claims to be."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"corrupt") + chunk(b"IEND", b"")


def test_valid_image(client):
    response = client.post("/detect", files={"image": ("canvas1.png", SAMPLE.read_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["boxes"]


def test_body_too_large(client):
    response = client.post("/detect", files={"image": ("big.png", b"\0" * (MAX_BODY_SIZE + 1), "image/png")})
    assert response.status_code == 413


def test_chunked_body_too_large(client):
    def body():  # a generator makes httpx send it chunked, without a Content-Length
        for _ in range(MAX_BODY_SIZE // 1024 + 2):
            yield b"\0" * 1024
    response = client.post("/detect", content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert response.status_code == 413


@pytest.mark.parametrize("side", [8000, 40000])  # over our limit, and over Pillow's own decompression bomb limit
def test_too_many_pixels(client, side):
    response = client.post("/detect", files={"image": ("bomb.png", png_header(side, side), "image/png")})
    assert response.status_code == 413


def test_not_an_image(client):
    response = client.post("/detect", files={"image": ("fake.png", b"definitely not a png", "image/png")})
    assert response.status_code == 400


@pytest.mark.parametrize("name, data", [
    ("empty", b""),
    ("webp", b"RIFF\x1a\x00\x00\x00WEBPVP8 \x0e\x00\x00\x00"),
    ("gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"),
    ("pdf", b"%PDF-1.4\n%%EOF"),
])
def test_unsupported_format(client, name, data):
    response = client.post("/detect", files={"image": (name, data, "image/png")})
    assert response.status_code == 400


@pytest.mark.parametrize("width, height", [(4000, 1), (1, 4000), (20000, 2)])
def test_extreme_aspect_ratio(client, width, height):
    image = cv2.imencode(".png", np.full((height, width), 255, np.uint8))[1].tobytes()
    response = client.post("/detect", files={"image": ("thin.png", image, "image/png")})
    assert response.status_code == 200


def test_undecodable_image(client):
    response = client.post("/detect", files={"image": ("broken.png", png_header(100, 100), "image/png")})
    assert response.status_code == 400
