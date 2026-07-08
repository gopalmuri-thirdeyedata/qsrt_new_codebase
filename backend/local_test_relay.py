"""
local_test_relay.py — local_relay.py pre-configured for local testing.

Reads from fake_camera.py instead of real RTSP cameras.
Sends to localhost FastAPI instead of the cloud server.

Usage:
  # Terminal 1 — start the FastAPI backend
  cd backend
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # Terminal 2 — start the fake camera server
  python fake_camera.py

  # Terminal 3 — start this relay
  python local_test_relay.py

  # Browser — open the frontend
  cd frontend && npm start
  # Then open http://localhost:3000
Install:
  pip install opencv-python-headless websockets flask numpy
"""

import asyncio
import cv2
import time
import logging
import websockets


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("relay-test")


# ── Local test config ─────────────────────────────────────────
CLOUD_WS_BASE  = "ws://localhost:8000"    # local FastAPI server


# Read from fake_camera.py MJPEG streams
CAMERAS = {
    # Tailscale IP of mobile (moto-g64-5g) — works across different networks!
    "station_1": "http://localhost:8554/station_1",
    "station_2": "http://localhost:8554/station_2",
    "station_3": "http://localhost:8554/station_3",
    "station_4": "http://localhost:8554/station_4",
}




TARGET_FPS    = 3
JPEG_QUALITY  = 65
SEND_SIZE     = (640, 480)
RECONNECT_DELAY = 3


async def relay_camera(station_id: str, stream_url: str):
    ws_url      = f"{CLOUD_WS_BASE}/ws/stream/{station_id}"
    frame_delay = 1.0 / TARGET_FPS
    log.info(f"[{station_id}] Starting → {ws_url}")

    while True:  # outer: reconnect loop
        try:
            async with websockets.connect(
                ws_url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                log.info(f"[{station_id}] Connected to server")

                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    log.error(f"[{station_id}] Cannot open stream {stream_url}")
                    log.error(f"[{station_id}] Is fake_camera.py running?")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue

                log.info(f"[{station_id}] Stream opened — sending at {TARGET_FPS} fps")
                frames_sent = 0

                while True:  # inner: frame streaming loop
                    t0 = time.monotonic()

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        log.warning(f"[{station_id}] Stream ended — reconnecting")
                        break

                    if SEND_SIZE:
                        frame = cv2.resize(frame, SEND_SIZE,
                                           interpolation=cv2.INTER_LINEAR)

                    _, buf = cv2.imencode(".jpg", frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    await ws.send(buf.tobytes())
                    frames_sent += 1

                    if frames_sent % 50 == 0:
                        log.info(f"[{station_id}] {frames_sent} frames sent")

                    elapsed = time.monotonic() - t0
                    sleep = frame_delay - elapsed
                    if sleep > 0:
                        await asyncio.sleep(sleep)

                cap.release()

        except Exception as e:
            log.error(f"[{station_id}] Error: {e} — retrying in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)


async def main():
    log.info("KitchEye local test relay starting")
    log.info(f"Server:  {CLOUD_WS_BASE}")
    log.info(f"Cameras: {list(CAMERAS.keys())}")
    await asyncio.gather(*[
        relay_camera(sid, url) for sid, url in CAMERAS.items()
    ])

if __name__ == "__main__":
    asyncio.run(main())
