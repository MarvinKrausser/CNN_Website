"""Routes: bird classification (POST /predict) and face detection
(websocket /predict_face)."""

from collections import defaultdict
import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from src.production import settings
from src.production.inference import InvalidImage, classify_bird, detect_faces
from src.production.limits import Busy, client_ip, gate, predict_limiter, rate_limit

logger = logging.getLogger("server")
router = APIRouter()


async def read_limited(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(64 * 1024):
        data += chunk
        if len(data) > limit:
            raise HTTPException(status_code=413, detail="File too large")
    return bytes(data)


@router.post("/predict", dependencies=[Depends(rate_limit(predict_limiter))])
async def predict(file: UploadFile = File(...)):
    data = await read_limited(file, settings.MAX_UPLOAD_BYTES)
    try:
        return await gate.run(classify_bird, data)
    except Busy:
        raise HTTPException(status_code=503, detail="Server busy, try again shortly", headers={"Retry-After": "5"})
    except InvalidImage as e:
        raise HTTPException(status_code=400, detail=str(e))


active_websockets = defaultdict(int)


@router.websocket("/predict_face")
async def predict_face(websocket: WebSocket):
    ip = client_ip(websocket)

    # CORS does not apply to websockets, so check the origin ourselves.
    if websocket.headers.get("origin") not in settings.ORIGINS:
        await websocket.close(code=1008)
        return
    if (sum(active_websockets.values()) >= settings.MAX_WEBSOCKETS
            or active_websockets[ip] >= settings.MAX_WEBSOCKETS_PER_IP):
        await websocket.close(code=1013)  # try again later
        return

    active_websockets[ip] += 1
    try:
        await websocket.accept()
        last = 0.0

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None or len(data) > settings.MAX_FRAME_BYTES:
                await websocket.close(code=1009 if data else 1003)
                break

            # Drop frames that arrive too fast or while the server is busy
            # instead of queueing them.
            now = time.monotonic()
            if now - last < settings.FACE_MIN_INTERVAL:
                continue
            last = now

            try:
                boxes = await gate.run(detect_faces, data)
            except Busy:
                continue
            except InvalidImage:
                await websocket.close(code=1003)
                break

            await websocket.send_json({"bboxes": boxes})

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("predict_face failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        active_websockets[ip] -= 1
        if active_websockets[ip] <= 0:
            del active_websockets[ip]
