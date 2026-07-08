import asyncio
import cv2
import base64
import json
import time
import uuid
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Food Detection API")

# Persistent storage: uploads/<video_id>/original.<ext> + uploads/<video_id>/processed.mp4
UPLOAD_ROOT = Path(__file__).parent / "uploads"
UPLOAD_ROOT.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("kitcheye")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    html_path = Path(__file__).parent.parent / "frontend" / "demoDashboard.html"
    return FileResponse(html_path)

@app.get("/monitor")
async def serve_monitor():
    html_path = Path(__file__).parent.parent / "frontend" / "live_monitor.html"
    return FileResponse(html_path)

# Serve stored uploads/processed videos back, e.g. /videos/<video_id>/processed.mp4
app.mount("/videos", StaticFiles(directory=str(UPLOAD_ROOT)), name="videos")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # Each upload gets its own folder under uploads/ holding both the
    # original file and (once processed) the annotated output video.
    video_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    video_dir = UPLOAD_ROOT / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    original_path = video_dir / f"original{ext}"
    with open(original_path, "wb") as out:
        # Stream read file in chunks of 1MB to prevent high memory usage
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    # Return absolute path so backend can read it directly
    return {"video_id": video_id, "temp_path": str(original_path)}


from qubeyond_integration import qubeyond_kitchen_event
app.add_api_route(
    "/webhook/qubeyond/kitchen-event",
    qubeyond_kitchen_event,
    methods=["POST"]
)

# Single-thread executor — MPS/CUDA are not thread-safe across workers
_executor = ThreadPoolExecutor(max_workers=1)

# Latest detection snapshot per station (for Qubeyond order verification)
_snapshots: dict = {}

# Browser viewer WebSocket connections — set of websockets per station_id
# Relay pushes frames → server processes → broadcasts to all viewers
_viewers: dict = {}   # station_id → set of WebSocket connections

# ── Tuning ────────────────────────────────────────────────────────────────────
FRAME_SKIP   = 1      # keep live stream processing on every frame
UPLOAD_FRAME_SKIP = 5 # sample uploaded videos for faster interactive feedback
INFER_SIZE   = 640    # input resolution fed to YOLO
JPEG_QUALITY = 75     # higher quality for visible detections
UPLOAD_INFER_SIZE = 416
UPLOAD_JPEG_QUALITY = 55
UPLOAD_PREVIEW_MAX_EDGE = 960
UPLOAD_CONFIDENCE = 0.5


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────



def load_model(model_path: str = "best.pt"):
    from ultralytics import YOLO
    import torch

    device = get_device()
    # Prefer custom best.pt, fall back to yolov8n.pt if missing
    custom_path = Path(__file__).parent / "best.pt"
    if custom_path.exists():
        path = str(custom_path)
    elif Path(model_path).exists():
        path = model_path
    else:
        path = "yolov8n.pt"

    model  = YOLO(path)
    model.to(device)

    if device == "cuda":
        model.model.half()   # FP16 on CUDA — ~2x faster

    # Warm-up: compile kernels before first real request
    dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, imgsz=INFER_SIZE, verbose=False)

    print(f"[kitcheye] YOLO ready on {device}: {path}")
    return model, device


_model  = None
_device = None

def get_model():
    return _model, _device

@app.on_event("startup")
async def startup_event():
    global _model, _device
    loop = asyncio.get_running_loop()
    _model, _device = await loop.run_in_executor(_executor, load_model)
    app.state.snapshots = _snapshots
    print("[kitcheye] Startup complete — ready to accept connections.")




# ── Frame helpers ─────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    h, w  = frame.shape[:2]
    scale = INFER_SIZE / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


def orient_for_detection(frame: np.ndarray) -> np.ndarray:
    """
    Uploaded phone/camera clips are often portrait or sideways.
    Rotate tall frames into landscape before inference so the model sees
    a more natural orientation and the UI updates in the same orientation.
    """
    h, w = frame.shape[:2]
    if h > w * 1.15:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def encode_frame(
    frame: np.ndarray,
    jpeg_quality: int = JPEG_QUALITY,
    max_edge: int | None = None,
) -> str:
    if max_edge:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest > max_edge:
            scale = max_edge / longest
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )

    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return base64.b64encode(buf).decode("utf-8")


def run_detection(
    model,
    frame: np.ndarray,
    conf: float = 0.15,
    infer_size: int = INFER_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    preview_max_edge: int | None = None,
):
    """YOLO inference + annotation. Returns (b64_jpeg, detections, annotated_frame)."""
    frame_for_detection = orient_for_detection(frame)
    small   = preprocess(frame_for_detection)
    results = model(small, conf=conf, verbose=False, imgsz=infer_size)[0]
    annotated = results.plot()   # draws boxes on the frame

    # Scale annotated frame back up to match original size for display
    h_orig, w_orig = frame_for_detection.shape[:2]
    if annotated.shape[:2] != (h_orig, w_orig):
        annotated = cv2.resize(annotated, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    boxes   = results.boxes
    if len(boxes) == 0:
        return encode_frame(annotated, jpeg_quality=jpeg_quality, max_edge=preview_max_edge), [], annotated

    # Vectorised extraction — no per-box Python loop
    xyxy    = boxes.xyxy.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    labels  = [model.names[i] for i in cls_ids]

    detections = [
        {
            "label":      str(labels[i]),
            "confidence": round(float(confs[i]), 2),
            "bbox":       [int(v) for v in xyxy[i]],  # plain Python ints
        }
        for i in range(len(labels))
    ]
    return encode_frame(annotated, jpeg_quality=jpeg_quality, max_edge=preview_max_edge), detections, annotated


# ── WebSocket: Video upload ───────────────────────────────────────────────────────────────

@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    """
    Supports both upload flows used by the frontend:
    1. A JSON text message containing {"temp_path": "..."} after /upload-video
    2. A single binary WebSocket payload containing the full video
    Streams annotated frames + detections with progress metadata.
    """
    await websocket.accept()
    model, _ = get_model()
    tmp_path = None
    video_id = None
    video_dir = None
    writer = None
    cap = None
    session_id = f"upload-{int(time.time() * 1000)}"

    try:
        first_message = await websocket.receive()

        if first_message.get("text") is not None:
            msg_data = json.loads(first_message["text"])
            tmp_path = msg_data.get("temp_path")
            if not tmp_path or not os.path.exists(tmp_path):
                raise FileNotFoundError(f"Temporary video file not found at: {tmp_path}")
            # /upload-video already saved this under uploads/<video_id>/original.*
            video_dir = Path(tmp_path).parent
            video_id = video_dir.name
        else:
            video_bytes = first_message.get("bytes")
            if not video_bytes:
                raise ValueError("Received empty video payload")

            video_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            video_dir = UPLOAD_ROOT / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = str(video_dir / "original.mp4")
            with open(tmp_path, "wb") as f:
                f.write(video_bytes)

        processed_path = video_dir / "processed.mp4"

        cap          = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Unable to open uploaded video")
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(1, (source_total_frames + UPLOAD_FRAME_SKIP - 1) // UPLOAD_FRAME_SKIP)
        read_q       = asyncio.Queue(maxsize=8)
        result_q     = asyncio.Queue(maxsize=8)
        start_time   = time.time()
        # Output video plays back at the sampled rate so its duration still
        # roughly tracks the source clip even though we only annotate every
        # UPLOAD_FRAME_SKIP-th frame.
        writer_fps   = max(1.0, fps / UPLOAD_FRAME_SKIP)

        log.info(
            "[%s] upload started: fps=%.2f source_frames=%s sampled_frames=%s skip=%s infer=%s conf=%.2f",
            session_id,
            fps,
            source_total_frames,
            total_frames,
            UPLOAD_FRAME_SKIP,
            UPLOAD_INFER_SIZE,
            UPLOAD_CONFIDENCE,
        )

        async def reader():
            idx = 0
            loop = asyncio.get_running_loop()
            while True:
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    await read_q.put(None)
                    break
                if idx % UPLOAD_FRAME_SKIP == 0:
                    await read_q.put((idx, frame.copy()))
                idx += 1

        async def detector():
            loop = asyncio.get_running_loop()
            processed = 0
            while True:
                item = await read_q.get()
                if item is None:
                    await result_q.put(None)
                    break
                f_idx, frame = item
                frame_started = time.time()
                img_b64, detections, annotated = await loop.run_in_executor(
                    _executor,
                    run_detection,
                    model,
                    frame,
                    UPLOAD_CONFIDENCE,
                    UPLOAD_INFER_SIZE,
                    UPLOAD_JPEG_QUALITY,
                    UPLOAD_PREVIEW_MAX_EDGE,
                )
                processed += 1
                infer_ms = round((time.time() - frame_started) * 1000)
                await result_q.put((f_idx, processed, img_b64, detections, infer_ms, annotated))

        async def sender():
            nonlocal writer
            while True:
                item = await result_q.get()
                if item is None:
                    break
                f_idx, processed, img_b64, detections, infer_ms, annotated = item

                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(processed_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        writer_fps,
                        (w, h),
                    )
                writer.write(annotated)

                elapsed_ms = round((time.time() - start_time) * 1000)
                processing_fps = round(processed / max(elapsed_ms / 1000, 0.001), 1)
                log.info(
                    "[%s] frame %s/%s source_idx=%s infer=%sms dets=%s stream_fps=%.2f",
                    session_id,
                    processed,
                    total_frames,
                    f_idx,
                    infer_ms,
                    len(detections),
                    processing_fps,
                )
                await websocket.send_text(json.dumps({
                    "frame":        img_b64,
                    "detections":   detections,
                    "frame_idx":    f_idx,
                    "processed_frames": processed,
                    "total_frames": total_frames,
                    "source_total_frames": source_total_frames,
                    "fps":          round(fps, 1),
                    "processing_fps": processing_fps,
                    "infer_ms":     infer_ms,
                    "elapsed_ms":   elapsed_ms,
                }))

        await asyncio.gather(reader(), detector(), sender())
        log.info("[%s] upload complete — stored at %s", session_id, video_dir)
        await websocket.send_text(json.dumps({
            "done": True,
            "video_id": video_id,
            "original_url": f"/videos/{video_id}/{Path(tmp_path).name}",
            "processed_url": f"/videos/{video_id}/processed.mp4" if writer is not None else None,
        }))

    except WebSocketDisconnect:
        log.warning("[%s] websocket disconnected", session_id)
    except Exception as e:
        log.exception("[%s] upload failed", session_id)
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        # Always finalize cap/writer — even on disconnect or a mid-stream
        # error — otherwise processed.mp4 is left without its moov atom
        # (unreadable/corrupt) and the source capture handle leaks.
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass


# ── WebSocket: Single live stream (frontend browser camera) ──────────────────

@app.websocket("/ws/stream")
async def stream_ws(websocket: WebSocket):
    """
    Client pushes JPEG frames as binary messages (one per frame).
    Server responds with annotated frame + detections.
    """
    await websocket.accept()
    model, _ = get_model()
    loop = asyncio.get_running_loop()

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            img_b64, detections, _ = await loop.run_in_executor(
                _executor, run_detection, model, frame
            )
            await websocket.send_text(json.dumps({
                "frame":      img_b64,
                "detections": detections,
                "timestamp":  time.time(),
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ── WebSocket: Multi-camera ingest (local_relay.py pushes frames here) ────────

@app.websocket("/ws/stream/{station_id}")
async def ingest_stream(websocket: WebSocket, station_id: str):
    """
    Receives JPEG frames pushed by local_relay.py from inside the restaurant.
    One persistent connection per camera station.
    Stores latest detections in _snapshots for Qubeyond order verification.
    """
    await websocket.accept()
    model, _ = get_model()
    loop = asyncio.get_running_loop()

    print(f"[{station_id}] Relay connected from {websocket.client.host}")

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            img_b64, detections, _ = await loop.run_in_executor(
                _executor, run_detection, model, frame
            )

            # Store for Qubeyond verification
            _snapshots[station_id] = {
                "detections": detections,
                "frame_b64":  img_b64,
                "timestamp":  time.time(),
            }

            # Broadcast to all browser viewers watching this station
            payload = json.dumps({
                "station_id": station_id,
                "frame":      img_b64,
                "detections": detections,
                "timestamp":  time.time(),
            })
            dead = set()
            for viewer in _viewers.get(station_id, set()):
                try:
                    await viewer.send_text(payload)
                except Exception:
                    dead.add(viewer)
            if dead:
                _viewers[station_id] -= dead

    except WebSocketDisconnect:
        print(f"[{station_id}] Relay disconnected")
    except Exception as e:
        print(f"[{station_id}] Error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ── WebSocket: Browser viewer (live-monitor.html connects here) ──────────────

@app.websocket("/ws/view/{station_id}")
async def view_stream(websocket: WebSocket, station_id: str):
    """
    Browser connects here to watch a station feed.
    Relay pushes frames to /ws/stream/{station_id},
    server processes them and broadcasts results to all viewers here.
    """
    await websocket.accept()

    # Register this viewer
    if station_id not in _viewers:
        _viewers[station_id] = set()
    _viewers[station_id].add(websocket)
    print(f"[{station_id}] Viewer connected ({len(_viewers[station_id])} total)")

    # Send the last known snapshot immediately so screen isn't blank
    snap = _snapshots.get(station_id)
    if snap:
        try:
            await websocket.send_text(json.dumps({
                "station_id": station_id,
                "frame":      snap["frame_b64"],
                "detections": snap["detections"],
                "timestamp":  snap["timestamp"],
            }))
        except Exception:
            pass

    try:
        # Keep connection alive — server pushes frames, viewer just receives
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        pass
    finally:
        _viewers.get(station_id, set()).discard(websocket)
        print(f"[{station_id}] Viewer disconnected")


@app.get("/ws/stations")
async def list_stations():
    """Lists which stations have active relays and viewers."""
    return {
        "relays":  list(_snapshots.keys()),
        "viewers": {sid: len(v) for sid, v in _viewers.items()},
    }


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    _, device = get_model()
    return {"status": "ok", "device": device, "model": "YOLO"}


@app.get("/model-info")
async def model_info():
    model, device = get_model()
    return {
        "device":      device,
        "classes":     model.names if model else {},
        "num_classes": len(model.names) if model else 0,
        "infer_size":  INFER_SIZE,
        "frame_skip":  FRAME_SKIP,
    }


@app.get("/stations")
async def get_stations():
    """Latest detection snapshot per station — used by Qubeyond integration."""
    return {
        sid: {
            "detections": snap["detections"],
            "age_seconds": round(time.time() - snap["timestamp"], 1),
        }
        for sid, snap in _snapshots.items()
    }
