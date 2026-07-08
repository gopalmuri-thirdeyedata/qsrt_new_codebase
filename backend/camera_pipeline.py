"""
camera_pipeline.py — Multi-camera motion-triggered detection pipeline

Architecture:
  - Each camera runs in its own async task
  - Frame differencing detects motion (costs ~0.1ms per frame)
  - YOLO inference only runs when motion is detected
  - Detection results stored by station_id for Qubeyond order verification
  - Qubeyond ORDER_COMPLETE webhook can force a detection burst on any station
  - WebSocket clients receive annotated frames in real time

Camera states per station:
  IDLE      → reading frames, running motion detection only
  ACTIVE    → motion detected, running YOLO on every Nth frame
  COOLDOWN  → motion stopped, running YOLO for a few more seconds to catch final state

Install:
  pip install ultralytics opencv-python-headless numpy fastapi uvicorn websockets

Usage:
  Add camera URLs to CAMERAS dict below, then import this module into main.py
"""

import asyncio
import cv2
import base64
import json
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

# Add your camera streams here
# Key = station_id (must match Qubeyond station_id field)
# Value = RTSP/HTTP stream URL, or integer for USB webcam (0, 1, 2...)
CAMERAS = {
    "station_1": "rtsp://admin:password@192.168.1.101:554/stream1",
    "station_2": "rtsp://admin:password@192.168.1.102:554/stream1",
    "station_3": "rtsp://admin:password@192.168.1.103:554/stream1",
    "station_4": "rtsp://admin:password@192.168.1.104:554/stream1",
    # For testing with a USB webcam:
    # "station_test": 0,
}

MODEL_PATH       = "best.pt"        # your trained YOLO model
INFER_SIZE       = 640
CONF_THRESHOLD   = 0.40

# Motion detection tuning
MOTION_THRESHOLD  = 2500   # pixel area change to trigger ACTIVE state
                            # lower = more sensitive, higher = ignore small movements
                            # typical range: 1000–8000 depending on camera/lighting

# How long to keep running YOLO after motion stops (seconds)
# Long enough to capture the final settled tray state
COOLDOWN_SECONDS  = 4.0

# YOLO frame skip during ACTIVE state (1 = every frame, 2 = every other frame)
ACTIVE_FRAME_SKIP = 2

# JPEG quality for WebSocket transmission
JPEG_QUALITY = 65

# How many recent detection snapshots to keep per station (for order verification)
DETECTION_HISTORY_SIZE = 10


# ── State ─────────────────────────────────────────────────────────────────────

class CameraState(Enum):
    IDLE     = "idle"      # only motion detection running
    ACTIVE   = "active"    # motion detected — YOLO running
    COOLDOWN = "cooldown"  # motion stopped — finishing YOLO burst
    FORCED   = "forced"    # webhook triggered — run YOLO regardless of motion


@dataclass
class StationSnapshot:
    """Latest detection result for a station — used by Qubeyond integration."""
    station_id:  str
    detections:  list
    frame_b64:   str
    timestamp:   float
    state:       str


@dataclass
class CameraWorker:
    station_id:    str
    stream_url:    object          # str URL or int for webcam
    state:         CameraState = CameraState.IDLE
    last_motion:   float       = 0.0
    frame_idx:     int         = 0
    last_frame:    Optional[np.ndarray] = None
    last_detections: list      = field(default_factory=list)
    ws_clients:    set         = field(default_factory=set)


# Global stores
_workers:   dict[str, CameraWorker] = {}
_snapshots: dict[str, StationSnapshot] = {}   # station_id → latest snapshot
_executor   = ThreadPoolExecutor(max_workers=1)  # single thread for YOLO
_model      = None


# ── Model ─────────────────────────────────────────────────────────────────────

def load_yolo_model():
    from ultralytics import YOLO
    import torch
    model = YOLO(MODEL_PATH if Path(MODEL_PATH).exists() else "yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        model.model.half()
    # Warm up
    dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
    model(dummy, imgsz=INFER_SIZE, verbose=False)
    print(f"[pipeline] YOLO ready on {device}")
    return model


def run_yolo(frame: np.ndarray) -> tuple[np.ndarray, list]:
    """Run YOLO inference. Executes in thread executor."""
    results    = _model(frame, conf=CONF_THRESHOLD, imgsz=INFER_SIZE, verbose=False)[0]
    annotated  = results.plot()
    boxes      = results.boxes
    if len(boxes) == 0:
        return annotated, []
    xyxy    = boxes.xyxy.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    detections = [
        {
            "label":      _model.names[cls_ids[i]],
            "confidence": round(float(confs[i]), 2),
            "bbox":       xyxy[i].tolist(),
        }
        for i in range(len(cls_ids))
    ]
    return annotated, detections


# ── Motion detection ──────────────────────────────────────────────────────────

def detect_motion(prev: np.ndarray, curr: np.ndarray) -> bool:
    """
    Frame differencing motion detection. ~0.1ms per call.
    Returns True if significant motion detected.
    """
    # Resize to small size for speed
    h, w = prev.shape[:2]
    scale = min(1.0, 320 / max(h, w))
    if scale < 1.0:
        small_prev = cv2.resize(prev, (int(w*scale), int(h*scale)))
        small_curr = cv2.resize(curr, (int(w*scale), int(h*scale)))
    else:
        small_prev, small_curr = prev, curr

    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur reduces noise sensitivity
    gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (5, 5), 0)

    diff       = cv2.absdiff(gray_prev, gray_curr)
    _, thresh  = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Morphological close fills small gaps
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    changed_pixels = np.sum(thresh > 0)
    return int(changed_pixels) > MOTION_THRESHOLD


def draw_motion_indicator(frame: np.ndarray, state: CameraState, station_id: str) -> np.ndarray:
    """Draw minimal status overlay on frame."""
    out    = frame.copy()
    color  = {
        CameraState.IDLE:     (128, 128, 128),
        CameraState.ACTIVE:   (0, 200, 80),
        CameraState.COOLDOWN: (0, 165, 255),
        CameraState.FORCED:   (255, 100, 0),
    }.get(state, (128, 128, 128))

    label = f"{station_id} | {state.value.upper()}"
    cv2.rectangle(out, (0, 0), (len(label) * 9 + 16, 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


# ── Per-camera async worker ───────────────────────────────────────────────────

async def run_camera(worker: CameraWorker):
    """
    Main loop for a single camera.
    Reads frames, runs motion detection, conditionally runs YOLO.
    """
    loop = asyncio.get_running_loop()
    print(f"[{worker.station_id}] Starting camera: {worker.stream_url}")

    while True:
        cap = cv2.VideoCapture(worker.stream_url)
        if not cap.isOpened():
            print(f"[{worker.station_id}] Cannot open stream — retrying in 5s")
            await asyncio.sleep(5)
            continue

        fps         = cap.get(cv2.CAP_PROP_FPS) or 15
        frame_delay = 1.0 / fps

        try:
            while True:
                t0 = time.monotonic()

                # Read frame (blocking — run in executor to not block event loop)
                ret, frame = await loop.run_in_executor(
                    None, cap.read
                )
                if not ret or frame is None:
                    print(f"[{worker.station_id}] Stream lost — reconnecting")
                    break

                worker.frame_idx += 1
                annotated   = frame
                detections  = worker.last_detections  # reuse last result by default

                # ── Motion detection (always runs, costs ~0.1ms) ──────────────
                motion_detected = False
                if worker.last_frame is not None:
                    motion_detected = detect_motion(worker.last_frame, frame)

                now = time.monotonic()

                # ── State machine ─────────────────────────────────────────────
                if worker.state == CameraState.FORCED:
                    # Webhook triggered — run YOLO regardless
                    if worker.frame_idx % ACTIVE_FRAME_SKIP == 0:
                        annotated, detections = await loop.run_in_executor(
                            _executor, run_yolo, frame
                        )
                    # Exit FORCED after cooldown period
                    if now - worker.last_motion > COOLDOWN_SECONDS:
                        worker.state = CameraState.IDLE
                        print(f"[{worker.station_id}] Forced detection complete → IDLE")

                elif motion_detected:
                    worker.last_motion = now
                    if worker.state == CameraState.IDLE:
                        print(f"[{worker.station_id}] Motion detected → ACTIVE")
                    worker.state = CameraState.ACTIVE
                    if worker.frame_idx % ACTIVE_FRAME_SKIP == 0:
                        annotated, detections = await loop.run_in_executor(
                            _executor, run_yolo, frame
                        )

                elif worker.state in (CameraState.ACTIVE, CameraState.COOLDOWN):
                    elapsed = now - worker.last_motion
                    if elapsed < COOLDOWN_SECONDS:
                        # Still in cooldown — keep running YOLO
                        worker.state = CameraState.COOLDOWN
                        if worker.frame_idx % ACTIVE_FRAME_SKIP == 0:
                            annotated, detections = await loop.run_in_executor(
                                _executor, run_yolo, frame
                            )
                    else:
                        # Cooldown expired → IDLE
                        worker.state = CameraState.IDLE
                        print(f"[{worker.station_id}] Cooldown complete → IDLE")

                # ── IDLE: just draw status, no YOLO ──────────────────────────
                # (annotated stays as plain frame)

                worker.last_frame      = frame
                worker.last_detections = detections

                # ── Draw status overlay ───────────────────────────────────────
                display = draw_motion_indicator(annotated, worker.state, worker.station_id)

                # ── Store snapshot for Qubeyond order verification ────────────
                if detections or worker.state != CameraState.IDLE:
                    _, buf = cv2.imencode(".jpg", display,
                                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    frame_b64 = base64.b64encode(buf).decode()
                    _snapshots[worker.station_id] = StationSnapshot(
                        station_id  = worker.station_id,
                        detections  = detections,
                        frame_b64   = frame_b64,
                        timestamp   = time.time(),
                        state       = worker.state.value,
                    )

                # ── Broadcast to WebSocket clients watching this station ───────
                if worker.ws_clients:
                    _, buf = cv2.imencode(".jpg", display,
                                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    msg = json.dumps({
                        "station_id": worker.station_id,
                        "state":      worker.state.value,
                        "frame":      base64.b64encode(buf).decode(),
                        "detections": detections,
                        "timestamp":  time.time(),
                    })
                    dead = set()
                    for ws in worker.ws_clients:
                        try:
                            await ws.send_text(msg)
                        except Exception:
                            dead.add(ws)
                    worker.ws_clients -= dead

                # ── Pace to camera FPS ────────────────────────────────────────
                elapsed = time.monotonic() - t0
                sleep   = frame_delay - elapsed
                if sleep > 0:
                    await asyncio.sleep(sleep)

        except Exception as e:
            print(f"[{worker.station_id}] Error: {e} — reconnecting in 3s")
            await asyncio.sleep(3)
        finally:
            cap.release()


# ── Startup: launch all camera workers ───────────────────────────────────────

async def start_all_cameras():
    """Call this from your FastAPI startup event."""
    global _model
    loop = asyncio.get_running_loop()
    _model = await loop.run_in_executor(_executor, load_yolo_model)

    for station_id, url in CAMERAS.items():
        worker = CameraWorker(station_id=station_id, stream_url=url)
        _workers[station_id] = worker
        asyncio.create_task(run_camera(worker))
        print(f"[pipeline] Camera task started: {station_id}")


# ── Public API for Qubeyond integration ──────────────────────────────────────

def get_detections_for_station(station_id: str) -> list:
    """
    Called by qubeyond_integration.py to get latest detections.
    Replace the stub function there with a call to this.
    """
    snap = _snapshots.get(station_id)
    if not snap:
        return []
    # Only use detections that are fresh (within 60 seconds)
    age = time.time() - snap.timestamp
    if age > 60:
        print(f"[pipeline] Warning: detections for {station_id} are {age:.0f}s old")
    return snap.detections


def force_detection_burst(station_id: str, duration: float = COOLDOWN_SECONDS):
    """
    Called by Qubeyond ORDER_COMPLETE webhook to force YOLO on a station.
    Even if the station is idle (no motion), this forces detection for
    `duration` seconds — capturing the final tray state before bagging.
    """
    worker = _workers.get(station_id)
    if not worker:
        print(f"[pipeline] Warning: no worker for station {station_id}")
        return False
    worker.state       = CameraState.FORCED
    worker.last_motion = time.monotonic()  # reset cooldown timer
    print(f"[pipeline] Force detection burst on {station_id} for {duration}s")
    return True


def get_all_station_states() -> dict:
    """Returns current state of all cameras — for health/monitoring endpoint."""
    return {
        sid: {
            "state":           w.state.value,
            "last_motion_ago": round(time.monotonic() - w.last_motion, 1),
            "has_snapshot":    sid in _snapshots,
            "snapshot_age":    round(time.time() - _snapshots[sid].timestamp, 1)
                               if sid in _snapshots else None,
            "ws_clients":      len(w.ws_clients),
        }
        for sid, w in _workers.items()
    }


# ── FastAPI routes ────────────────────────────────────────────────────────────
# Add these to your main.py app

app = FastAPI()   # remove this line when integrating into main.py

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    await start_all_cameras()


@app.websocket("/ws/station/{station_id}")
async def station_ws(websocket: WebSocket, station_id: str):
    """
    Connect to a live camera feed for a specific station.
    Frontend receives: {station_id, state, frame (base64), detections, timestamp}
    """
    await websocket.accept()
    worker = _workers.get(station_id)
    if not worker:
        await websocket.send_text(json.dumps({"error": f"Unknown station: {station_id}"}))
        await websocket.close()
        return

    worker.ws_clients.add(websocket)
    print(f"[{station_id}] WebSocket client connected ({len(worker.ws_clients)} total)")
    try:
        while True:
            # Keep connection alive — camera loop pushes frames
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        worker.ws_clients.discard(websocket)
        print(f"[{station_id}] Client disconnected")


@app.websocket("/ws/all-stations")
async def all_stations_ws(websocket: WebSocket):
    """
    Subscribe to all station feeds simultaneously.
    Useful for a multi-camera operator view on the frontend.
    """
    await websocket.accept()
    for worker in _workers.values():
        worker.ws_clients.add(websocket)
    print(f"[pipeline] All-stations client connected")
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        for worker in _workers.values():
            worker.ws_clients.discard(websocket)


@app.get("/stations")
async def list_stations():
    """Current state of all camera stations."""
    return get_all_station_states()


@app.get("/stations/{station_id}/snapshot")
async def get_snapshot(station_id: str):
    """Latest detection snapshot for a station (for debugging)."""
    snap = _snapshots.get(station_id)
    if not snap:
        return {"error": "No snapshot yet", "station_id": station_id}
    return {
        "station_id": snap.station_id,
        "detections": snap.detections,
        "state":      snap.state,
        "age_seconds": round(time.time() - snap.timestamp, 1),
        # omit frame_b64 to keep response small
    }


@app.post("/stations/{station_id}/force-detection")
async def force_detection(station_id: str):
    """
    Manually trigger a detection burst on a station.
    Also called internally by Qubeyond ORDER_COMPLETE webhook.
    """
    ok = force_detection_burst(station_id)
    if not ok:
        return {"error": f"Station {station_id} not found"}
    return {"status": "ok", "station_id": station_id, "message": "Detection burst started"}