"""
local_relay.py — KitchEye Restaurant Camera Relay

Runs INSIDE the restaurant on any local PC (Windows, Linux, Raspberry Pi).
Reads physical IP camera RTSP streams over the restaurant LAN and streams
compressed JPEG frames to the KitchEye cloud server over WebSocket.

Why this exists:
  IP cameras use RTSP, which requires direct LAN access.
  The cloud server cannot reach cameras behind a restaurant router/firewall.
  This relay bridges the gap: it runs on-site, reads cameras locally,
  and forwards frames outward over a standard outbound HTTPS/WSS connection.

Requirements:
  - Python 3.10+
  - pip install opencv-python-headless websockets python-dotenv

Configuration:
  Set environment variables in backend/.env:
    CLOUD_WS_BASE=wss://your-cloud-server.com
    CAM_1=rtsp://admin:password@192.168.1.101:554/stream1
    CAM_2=rtsp://admin:password@192.168.1.102:554/stream1
    CAM_3=rtsp://admin:password@192.168.1.103:554/stream1
    CAM_4=rtsp://admin:password@192.168.1.104:554/stream1

Run:
  python local_relay.py

Run as a background service (Linux / Raspberry Pi):
  See systemd instructions at the bottom of this file.

Run as a background service (Windows):
  See Windows Task Scheduler / NSSM instructions at the bottom of this file.
"""

# ── Standard Library Imports ──────────────────────────────────────────────────
import asyncio    # Concurrent async tasks for running all camera relays simultaneously
import cv2        # OpenCV: read RTSP streams, resize frames, encode to JPEG
import time       # Monotonic clock for FPS pacing and throughput logging
import logging    # Structured console logging with timestamps
import os         # Read environment variables
from dotenv import load_dotenv  # Load configuration from backend/.env file


# ── Environment & Logging Setup ───────────────────────────────────────────────

load_dotenv()  # Load CLOUD_WS_BASE, CAM_1..4 from backend/.env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("relay")


# ── Configuration ─────────────────────────────────────────────────────────────

# WebSocket base URL of the cloud FastAPI server.
# All station streams connect to: {CLOUD_WS_BASE}/ws/stream/{station_id}
# Examples:
#   wss://c2lyv8int2fk1m-8000.proxy.runpod.net   (RunPod)
#   wss://your-lambda-ip:8000                     (Lambda Labs)
#   wss://yourdomain.com                          (custom domain with SSL)
CLOUD_WS_BASE = os.getenv("CLOUD_WS_BASE", "wss://your-cloud-server.com")

# Camera station map for this restaurant location.
# Key   = station_id — must match what Qubeyond sends in the station_id field
#         and what the cloud server uses in /ws/stream/{station_id}
# Value = RTSP URL (IP camera) or integer (0, 1, 2 for USB/built-in webcam)
CAMERAS = {
    "station_1": os.getenv("CAM_1", "rtsp://admin:password@192.168.1.101:554/stream1"),
    "station_2": os.getenv("CAM_2", "rtsp://admin:password@192.168.1.102:554/stream1"),
    "station_3": os.getenv("CAM_3", "rtsp://admin:password@192.168.1.103:554/stream1"),
    "station_4": os.getenv("CAM_4", "rtsp://admin:password@192.168.1.104:554/stream1"),
}

# Target frame rate to send to the cloud server.
# 5 FPS is enough for food order verification — food doesn't move fast.
# Lower value = less bandwidth used, higher value = more responsive detection.
TARGET_FPS = 5

# JPEG compression quality for frames sent over the network.
# Range: 1 (smallest file, worst quality) to 100 (largest file, best quality).
# 50–65 is a good balance for reliable detection with minimal bandwidth.
JPEG_QUALITY = 60

# Resize frames to this resolution before JPEG encoding and sending.
# Reduces bandwidth significantly with minimal impact on YOLO detection accuracy.
# Set to None to send original camera resolution (higher bandwidth).
SEND_SIZE = (640, 480)

# Seconds to wait before retrying after a connection or camera failure.
RECONNECT_DELAY = 3


# ── Per-Camera Relay Coroutine ────────────────────────────────────────────────

async def relay_camera(station_id: str, stream_url):
    """
    Continuously read frames from one camera and stream them to the cloud server.

    Connects to the cloud WebSocket at /ws/stream/{station_id}.
    Reads the camera at TARGET_FPS, compresses each frame to JPEG,
    and sends the raw bytes over the WebSocket.

    Automatically reconnects on any failure:
      - WebSocket connection drops
      - Camera RTSP stream fails or camera goes offline
      - Network error

    Args:
        station_id:  Unique camera identifier (e.g. "station_1").
                     Must match the station_id Qubeyond uses in order events.
        stream_url:  RTSP URL string or integer webcam index.
    """
    import websockets  # Imported here to keep the top-level imports minimal

    # Target WebSocket URL on the cloud server for this station
    ws_url      = f"{CLOUD_WS_BASE}/ws/stream/{station_id}"
    # Time to sleep between frames to maintain the target FPS
    frame_delay = 1.0 / TARGET_FPS

    log.info(f"[{station_id}] Starting relay → {ws_url}")

    while True:
        # Outer reconnect loop — any failure restarts from here after RECONNECT_DELAY
        try:
            async with websockets.connect(
                ws_url,
                max_size=10 * 1024 * 1024,  # Allow up to 10 MB per message
                ping_interval=20,            # Send WebSocket pings every 20s to keep connection alive
                ping_timeout=10,             # Close connection if ping is not acknowledged in 10s
            ) as ws:
                log.info(f"[{station_id}] Connected to cloud server")

                # Open the camera stream (RTSP URL or webcam index)
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    log.error(f"[{station_id}] Cannot open camera — retrying in {RECONNECT_DELAY}s")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue

                # Limit OpenCV's internal frame buffer to 1 frame so we always
                # get the latest available frame, not a buffered old one
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                log.info(f"[{station_id}] Camera opened — streaming at {TARGET_FPS} fps")

                frames_sent = 0
                t_start     = time.monotonic()  # Start time for throughput calculation

                # Inner frame loop — runs for as long as the camera is available
                while True:
                    t0 = time.monotonic()  # Timestamp before this frame cycle

                    # Read one frame from the camera
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # Camera lost signal or stream ended — trigger reconnect
                        log.warning(f"[{station_id}] Frame read failed — reconnecting camera")
                        break

                    # Resize frame to reduce bandwidth before encoding
                    if SEND_SIZE:
                        frame = cv2.resize(frame, SEND_SIZE, interpolation=cv2.INTER_LINEAR)

                    # Compress frame to JPEG bytes in memory (not saved to disk)
                    _, buf = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )

                    # Send raw JPEG bytes to the cloud server.
                    # The /ws/stream/{station_id} endpoint expects binary JPEG data.
                    await ws.send(buf.tobytes())
                    frames_sent += 1

                    # Log throughput stats every 100 frames for monitoring
                    if frames_sent % 100 == 0:
                        elapsed      = time.monotonic() - t_start
                        actual_fps   = frames_sent / elapsed
                        kb_per_frame = len(buf) / 1024
                        log.info(
                            f"[{station_id}] {frames_sent} frames sent | "
                            f"{actual_fps:.1f} fps | "
                            f"~{kb_per_frame:.0f} KB/frame | "
                            f"~{kb_per_frame * actual_fps / 1024:.1f} MB/s"
                        )

                    # Rate limiting: sleep the remaining time in this frame's time slot
                    # to maintain TARGET_FPS without busy-waiting
                    elapsed = time.monotonic() - t0
                    sleep   = frame_delay - elapsed
                    if sleep > 0:
                        await asyncio.sleep(sleep)

                # Release camera handle before reconnecting
                cap.release()

        except Exception as e:
            # WebSocket error, network issue, or any unexpected failure
            log.error(f"[{station_id}] Connection error: {e} — retrying in {RECONNECT_DELAY}s")
            await asyncio.sleep(RECONNECT_DELAY)


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    """
    Start all configured camera relays concurrently.
    Each camera runs as an independent async task with its own reconnect loop.
    """
    log.info("KitchEye local relay starting")
    log.info(f"Cloud server: {CLOUD_WS_BASE}")
    log.info(f"Cameras: {list(CAMERAS.keys())}")
    log.info(f"Streaming at {TARGET_FPS} fps, JPEG quality {JPEG_QUALITY}")

    # Launch one relay_camera coroutine per configured camera station
    tasks = [
        relay_camera(station_id, url)
        for station_id, url in CAMERAS.items()
    ]
    await asyncio.gather(*tasks)  # Run all cameras concurrently, indefinitely


if __name__ == "__main__":
    asyncio.run(main())


# ════════════════════════════════════════════════════════════════════════════════
# RUNNING AS A BACKGROUND SERVICE — LINUX / RASPBERRY PI (systemd)
# ════════════════════════════════════════════════════════════════════════════════
#
# 1. Create the service file:
#    sudo nano /etc/systemd/system/kitcheye-relay.service
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
#
# [Install]
# WantedBy=multi-user.target
#
# 2. Enable and start:
#    sudo systemctl enable kitcheye-relay
#    sudo systemctl start kitcheye-relay
#    sudo systemctl status kitcheye-relay   # check it's running
#    journalctl -u kitcheye-relay -f        # watch live logs
#
# ════════════════════════════════════════════════════════════════════════════════
# RUNNING AS A BACKGROUND SERVICE — WINDOWS (Task Scheduler or NSSM)
# ════════════════════════════════════════════════════════════════════════════════
#
# Option A — Task Scheduler (no extra tools):
#   Task Scheduler → Create Task
#   Triggers: At startup
#   Action:   python C:\kitcheye\local_relay.py
#
# Option B — NSSM (Non-Sucking Service Manager, recommended):
#   nssm install KitchEyeRelay python C:\kitcheye\local_relay.py
#   nssm start KitchEyeRelay
#   nssm status KitchEyeRelay
#
# ════════════════════════════════════════════════════════════════════════════════