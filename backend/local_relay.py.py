"""
local_relay.py — Runs INSIDE the restaurant on a local machine.

Reads RTSP camera feeds and streams compressed JPEG frames to
your cloud FastAPI server over WebSocket.

Requirements:
  - Any machine on the same LAN as the cameras
  - Python 3.10+
  - Outbound internet access (just port 443 / WSS)

Install:
  pip install opencv-python-headless websockets python-dotenv

Run:
  python local_relay.py

Run as a background service (Linux):
  See bottom of this file for systemd instructions.
"""

import asyncio
import cv2
import base64
import json
import time
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("relay")

# ── Config ────────────────────────────────────────────────────────────────────

# Your cloud FastAPI server WebSocket base URL
# e.g. wss://c2lyv8int2fk1m-8000.proxy.runpod.net  (RunPod)
#      wss://your-lambda-ip:8000                    (Lambda Labs)
#      wss://yourdomain.com                         (custom domain)
CLOUD_WS_BASE = os.getenv("CLOUD_WS_BASE", "wss://your-cloud-server.com")

# Cameras in this restaurant
# Key   = station_id (must match CAMERAS dict in camera_pipeline.py on the server)
# Value = RTSP URL or integer (0,1,2 for USB webcams)
CAMERAS = {
    "station_1": os.getenv("CAM_1", "rtsp://admin:password@192.168.1.101:554/stream1"),
    "station_2": os.getenv("CAM_2", "rtsp://admin:password@192.168.1.102:554/stream1"),
    "station_3": os.getenv("CAM_3", "rtsp://admin:password@192.168.1.103:554/stream1"),
    "station_4": os.getenv("CAM_4", "rtsp://admin:password@192.168.1.104:554/stream1"),
}

# How many frames per second to send to the cloud
# Lower = less bandwidth, higher latency between detections
# 5 fps is enough for order verification — food doesn't move fast
TARGET_FPS   = 5
# JPEG compression quality sent over the wire
# 50–65 is a good balance of quality vs bandwidth
JPEG_QUALITY = 60

# Resize frames before sending — reduces bandwidth significantly
# None = send original resolution, (640, 480) = resize to this
SEND_SIZE = (640, 480)

# Reconnect delay if connection drops (seconds)
RECONNECT_DELAY = 3


# ── Per-camera relay coroutine ────────────────────────────────────────────────

async def relay_camera(station_id: str, stream_url):
    """
    Continuously reads one camera and streams frames to the cloud server.
    Automatically reconnects on failure.
    """
    import websockets

    ws_url      = f"{CLOUD_WS_BASE}/ws/stream/{station_id}"
    frame_delay = 1.0 / TARGET_FPS

    log.info(f"[{station_id}] Starting relay → {ws_url}")

    while True:  # outer loop: reconnect on any failure
        try:
            async with websockets.connect(
                ws_url,
                max_size=10 * 1024 * 1024,   # 10 MB max message
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                log.info(f"[{station_id}] Connected to cloud server")

                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    log.error(f"[{station_id}] Cannot open camera — retrying in {RECONNECT_DELAY}s")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue

                # Set camera buffer to 1 frame to always get the latest frame
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                log.info(f"[{station_id}] Camera opened — streaming at {TARGET_FPS} fps")
                frames_sent = 0
                t_start     = time.monotonic()

                while True:
                    t0 = time.monotonic()

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        log.warning(f"[{station_id}] Frame read failed — reconnecting camera")
                        break

                    # Resize to reduce bandwidth
                    if SEND_SIZE:
                        frame = cv2.resize(frame, SEND_SIZE, interpolation=cv2.INTER_LINEAR)

                    # Encode to JPEG
                    _, buf = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )

                    # Send raw JPEG bytes — server's /ws/stream/{station_id} expects this
                    await ws.send(buf.tobytes())
                    frames_sent += 1

                    # Log throughput every 100 frames
                    if frames_sent % 100 == 0:
                        elapsed = time.monotonic() - t_start
                        actual_fps = frames_sent / elapsed
                        kb_per_frame = len(buf) / 1024
                        log.info(
                            f"[{station_id}] {frames_sent} frames sent | "
                            f"{actual_fps:.1f} fps | "
                            f"~{kb_per_frame:.0f} KB/frame | "
                            f"~{kb_per_frame * actual_fps / 1024:.1f} MB/s"
                        )

                    # Pace to TARGET_FPS
                    elapsed = time.monotonic() - t0
                    sleep   = frame_delay - elapsed
                    if sleep > 0:
                        await asyncio.sleep(sleep)

                cap.release()

        except Exception as e:
            log.error(f"[{station_id}] Connection error: {e} — retrying in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)


# ── Cloud server: add this endpoint to camera_pipeline.py ────────────────────
# The relay sends raw JPEG bytes per frame.
# Add this WebSocket endpoint to your cloud FastAPI app so the relay
# can push frames to the right station worker:
#
# @app.websocket("/ws/stream/{station_id}")
# async def ingest_stream(websocket: WebSocket, station_id: str):
#     await websocket.accept()
#     worker = _workers.get(station_id)
#     if not worker:
#         await websocket.close()
#         return
#     loop = asyncio.get_running_loop()
#     try:
#         while True:
#             frame_bytes = await websocket.receive_bytes()
#             arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
#             frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#             if frame is None:
#                 continue
#             # Inject frame into the camera worker
#             # (bypasses cap.read() — worker processes this frame instead)
#             worker.last_frame = frame
#             # Trigger motion detection and YOLO as normal
#             asyncio.create_task(_process_injected_frame(worker, frame, loop))
#     except WebSocketDisconnect:
#         pass


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    log.info(f"KitchEye local relay starting")
    log.info(f"Cloud server: {CLOUD_WS_BASE}")
    log.info(f"Cameras: {list(CAMERAS.keys())}")
    log.info(f"Streaming at {TARGET_FPS} fps, JPEG quality {JPEG_QUALITY}")

    # Start all camera relays concurrently
    tasks = [
        relay_camera(station_id, url)
        for station_id, url in CAMERAS.items()
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())


# ════════════════════════════════════════════════════════════════════════════
# RUNNING AS A BACKGROUND SERVICE (Linux / Raspberry Pi)
# ════════════════════════════════════════════════════════════════════════════
#
# Create /etc/systemd/system/kitcheye-relay.service:
#
# [Unit]
# Description=KitchEye Camera Relay
# After=network-online.target
# Wants=network-online.target
#
# [Service]
# Type=simple
# User=pi
# WorkingDirectory=/home/pi/kitcheye
# ExecStart=/home/pi/kitcheye/venv/bin/python local_relay.py
# Restart=always
# RestartSec=5
# Environment=CLOUD_WS_BASE=wss://your-cloud-server.com
# Environment=CAM_1=rtsp://admin:password@192.168.1.101:554/stream1
# Environment=CAM_2=rtsp://admin:password@192.168.1.102:554/stream1
# Environment=CAM_3=rtsp://admin:password@192.168.1.103:554/stream1
# Environment=CAM_4=rtsp://admin:password@192.168.1.104:554/stream1
#
# [Install]
# WantedBy=multi-user.target
#
# Then run:
#   sudo systemctl enable kitcheye-relay
#   sudo systemctl start kitcheye-relay
#   sudo systemctl status kitcheye-relay   # check it's running
#   journalctl -u kitcheye-relay -f        # watch live logs
#
# ════════════════════════════════════════════════════════════════════════════
# RUNNING ON WINDOWS (restaurant PC)
# ════════════════════════════════════════════════════════════════════════════
#
# Create a scheduled task to run on startup:
#   Task Scheduler → Create Task → Triggers: At startup
#   Action: python C:\kitcheye\local_relay.py
#
# Or use NSSM (Non-Sucking Service Manager) to install as a Windows service:
#   nssm install KitchEyeRelay python C:\kitcheye\local_relay.py
#   nssm start KitchEyeRelay