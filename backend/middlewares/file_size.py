from http import HTTPStatus

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class MaxBodySizeMiddleware:
    """Rejects request bodies over max_size while they are being received.

    FastAPI parses the whole multipart body before the route runs, so a check in the route only fires after
    an arbitrarily large upload has already been spooled to disk."""

    def __init__(self, app: ASGIApp, max_size: int):
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > self.max_size:
            response = JSONResponse({"detail": "Request body too large"}, status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return await response(scope, receive, send)

        # I don't trust Content-Lenght. Verify it
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_size:
                    raise HTTPException(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE, detail="Request body too large")
            return message

        await self.app(scope, limited_receive, send)