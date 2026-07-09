"""
main.py — KitchEye FastAPI Backend Server

This is the central server for the KitchEye food detection system.
It handles:
  - Serving the HTML frontend dashboards (demoDashboard & live_monitor)
  - Loading and running the YOLOv8 food detection model
  - Processing uploaded video files with frame-by-frame YOLO annotation
  - Receiving live JPEG frames from local_relay.py via WebSocket
  - Broadcasting annotated frames to browser dashboard viewers
  - Exposing REST endpoints for health checks and station snapshots
  - Routing the Qubeyond POS webhook to qubeyond_integration.py

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

# ── Standard Library Imports ──────────────────────────────────────────────────
import asyncio                          # Async task management and event loop
import cv2                              # OpenCV: frame decoding, resizing, encoding, video writing
import base64                           # Encode binary image frames to Base64 text for JSON transport
import json                             # Serialize/deserialize data for WebSocket messages
import time                             # Timestamps for snapshots and FPS calculations
import uuid                             # Generate unique IDs for uploaded video sessions
import logging                          # Structured console logging with timestamps
import os                               # File system and environment variable access
import numpy as np                      # NumPy: image frames stored as multi-dimensional arrays
from pathlib import Path                # Clean, OS-independent file path handling
from concurrent.futures import ThreadPoolExecutor  # Run blocking YOLO inference in a background thread

# ── FastAPI Imports ───────────────────────────────────────────────────────────
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # Allow browser cross-origin requests
from fastapi.responses import FileResponse          # Send HTML files as HTTP responses
from fastapi.staticfiles import StaticFiles         # Serve uploaded video files as static assets


# ── App Initialization ────────────────────────────────────────────────────────

app = FastAPI(title="KitchEye Food Detection API")

# Root directory for all uploaded and processed video files
# Structure: uploads/<video_id>/original.<ext>  and  uploads/<video_id>/processed.mp4
UPLOAD_ROOT = Path(__file__).parent / "uploads"
UPLOAD_ROOT.mkdir(exist_ok=True)  # Create the folder on startup if it doesn't exist

# Configure application-wide logging with timestamps and log level labels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("kitcheye")


# ── CORS Middleware ───────────────────────────────────────────────────────────
# Allow all origins so the frontend (on any port or domain) can call this API.
# In production, replace allow_origins=["*"] with your specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Frontend Page Routes ──────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the main operator/analytics dashboard (demoDashboard.html)."""
    html_path = Path(__file__).parent.parent / "frontend" / "demoDashboard.html"
    return FileResponse(html_path)


@app.get("/monitor")
async def serve_monitor():
    """Serve the live multi-station camera monitor (live_monitor.html)."""
    html_path = Path(__file__).parent.parent / "frontend" / "live_monitor.html"
    return FileResponse(html_path)


# Expose the uploads/ directory at /videos/ so the frontend can stream processed videos.
# Example: /videos/<video_id>/processed.mp4
app.mount("/videos", StaticFiles(directory=str(UPLOAD_ROOT)), name="videos")


# ── Video Upload REST Endpoint ────────────────────────────────────────────────

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accept a video file upload from the frontend.

    Each upload is saved to its own unique folder under uploads/.
    Returns the video_id and the saved file path so the client
    can then open /ws/video and reference this file for processing.
    """
    # Create a unique folder name: timestamp_ms + 8 random hex chars
    # Example: 1720516740000_3a5b6c7d
    video_id  = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    video_dir = UPLOAD_ROOT / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the original file extension (.mp4, .mov, etc.)
    ext           = Path(file.filename or "video.mp4").suffix or ".mp4"
    original_path = video_dir / f"original{ext}"

    # Write the uploaded file in 1 MB chunks to avoid loading it all into memory
    with open(original_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    # Return the path so /ws/video can locate and open the file
    return {"video_id": video_id, "temp_path": str(original_path)}


# ── Qubeyond Webhook Route ────────────────────────────────────────────────────
# Import the handler from its dedicated module and register it as a POST route.
# Qubeyond calls this endpoint when an ORDER_COMPLETE event fires.
from qubeyond_integration import qubeyond_kitchen_event
app.add_api_route(
    "/webhook/qubeyond/kitchen-event",
    qubeyond_kitchen_event,
    methods=["POST"]
)


# ── Global State ──────────────────────────────────────────────────────────────

# Single-thread executor for YOLO inference.
# GPU (CUDA) and Apple Silicon (MPS) are NOT thread-safe — a single worker
# serializes all inference calls to prevent crashes or race conditions.
_executor = ThreadPoolExecutor(max_workers=1)

# Latest detection snapshot per camera station.
# Key   = station_id (e.g. "station_1")
# Value = {"detections": [...], "frame_b64": "...", "timestamp": float}
# Updated every frame by ingest_stream; read by qubeyond_integration for order verification.
_snapshots: dict = {}

# Active browser viewer WebSocket connections, grouped by station.
# Key   = station_id
# Value = set of active WebSocket objects
# When ingest_stream processes a frame, it fans it out to every viewer in the set.
_viewers: dict = {}


# ── Inference Tuning Constants ────────────────────────────────────────────────

# Live relay stream: process every single frame for lowest latency
FRAME_SKIP   = 1

# Uploaded video: sample every 5th frame — faster feedback, less GPU load
UPLOAD_FRAME_SKIP = 5

# YOLO input resolution for live streams (pixels on the longest edge)
INFER_SIZE   = 640

# JPEG quality for broadcast frames (higher = clearer bounding boxes)
JPEG_QUALITY = 75

# Reduced settings for uploaded videos (faster processing, smaller preview images)
UPLOAD_INFER_SIZE        = 416   # Smaller input → faster inference
UPLOAD_JPEG_QUALITY      = 55    # Lower quality → smaller WS messages
UPLOAD_PREVIEW_MAX_EDGE  = 960   # Cap preview image longest edge at 960px
UPLOAD_CONFIDENCE        = 0.5   # Higher confidence threshold for upload mode


# ── Hardware Device Detection ─────────────────────────────────────────────────

def get_device() -> str:
    """
    Detect the best available compute device for YOLO inference.
    Priority order: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass  # torch not installed — fall through to CPU
    return "cpu"


# ── YOLO Model Loading ────────────────────────────────────────────────────────

def load_model(model_path: str = "best.pt"):
    """
    Load the YOLOv8 model and move it to the best available device.

    Model file priority:
      1. backend/best.pt   — your custom trained model
      2. model_path param  — explicit path provided
      3. yolov8n.pt        — auto-downloaded YOLOv8 nano fallback

    On CUDA: enables FP16 (half precision) for ~2x faster inference.
    Runs 3 warm-up inferences so GPU kernels are compiled before real requests arrive.
    """
    from ultralytics import YOLO
    import torch

    device = get_device()

    # Determine which model file to load
    custom_path   = Path(__file__).parent / "model" / "best.pt"
    fallback_path = Path(__file__).parent / "model" / "yolov8n.pt"
    
    if custom_path.exists():
        path = str(custom_path)          # Prefer the custom trained model in models folder
    elif fallback_path.exists():
        path = str(fallback_path)        # Use local nano model if present
    elif Path(model_path).exists():
        path = model_path                # Use explicitly provided path
    else:
        path = str(fallback_path)        # Let YOLO download to model/yolov8n.pt directly

    model = YOLO(path)
    model.to(device)

    # FP16 on CUDA halves memory usage and roughly doubles throughput
    if device == "cuda":
        model.model.half()

    # Warm-up: run inference 3 times on a dummy black frame to pre-compile
    # GPU shaders and JIT code — prevents a slow first real request
    dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
    for _ in range(3):
        model(dummy, imgsz=INFER_SIZE, verbose=False)

    print(f"[kitcheye] YOLO ready on {device}: {path}")
    return model, device


# Module-level globals; populated during startup_event
_model  = None
_device = None


def get_model():
    """Return the loaded YOLO model and device string. Called by all inference paths."""
    return _model, _device


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup hook — runs once when the server boots.

    Loads the YOLO model in a background thread (load_model is CPU/GPU intensive
    and would block the async event loop if called directly).
    Also shares the _snapshots dict with qubeyond_integration via app.state.
    """
    global _model, _device
    loop = asyncio.get_running_loop()

    # Run blocking model load in the background thread pool
    _model, _device = await loop.run_in_executor(_executor, load_model)

    # Share snapshots with qubeyond_integration so it can read latest detections
    app.state.snapshots = _snapshots

    print("[kitcheye] Startup complete — ready to accept connections.")


# ── Frame Processing Helpers ──────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Resize a frame so its longest side equals INFER_SIZE (640px).
    Maintains aspect ratio. Returns unchanged if frame is already small enough.
    """
    h, w  = frame.shape[:2]
    scale = INFER_SIZE / max(h, w)
    if scale >= 1:
        return frame  # No upscaling needed
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


def orient_for_detection(frame: np.ndarray) -> np.ndarray:
    """
    Rotate portrait-mode frames to landscape before YOLO inference.

    Phone-recorded videos are often taller than wide. YOLO was trained on
    landscape images, so rotating portrait frames improves detection accuracy.
    Only rotates if height > 1.15 × width (clearly portrait).
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
    """
    Compress an OpenCV BGR frame to JPEG and return it as a Base64 string.

    Base64 encoding is required to embed binary image data inside JSON messages
    sent over WebSockets.

    Args:
        frame:        OpenCV NumPy image array (BGR).
        jpeg_quality: JPEG compression level (1–100). Higher = better quality, larger file.
        max_edge:     If set, downscale the frame so its longest side ≤ max_edge pixels.
    """
    # Optional: downscale frame for smaller WebSocket payload
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

    # Encode to in-memory JPEG bytes, then convert to Base64 text
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return base64.b64encode(buf).decode("utf-8")


def run_detection(
    model,
    frame: np.ndarray,
    conf: float = 0.15,
    infer_size: int = INFER_SIZE,
    jpeg_quality: int = JPEG_QUALITY,
    preview_max_edge: int | None = None,
):
    """
    Run YOLOv8 inference on a single frame and return results.

    Steps:
      1. Orient frame to landscape if portrait.
      2. Resize to inference resolution.
      3. Run YOLO model — returns bounding boxes, class IDs, confidence scores.
      4. Draw annotated boxes on the frame (results.plot()).
      5. Scale annotated frame back to original size.
      6. Build a structured list of detections.
      7. Encode the annotated frame as Base64 JPEG.

    Returns:
        Tuple of (base64_jpeg_string, detections_list, annotated_frame_array)
        detections_list format: [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
    """
    # Step 1 & 2: Orient and resize
    frame_for_detection = orient_for_detection(frame)
    small   = preprocess(frame_for_detection)

    # Step 3: Run YOLO inference
    results   = model(small, conf=conf, verbose=False, imgsz=infer_size)[0]

    # Step 4: Draw bounding boxes and labels on the frame
    annotated = results.plot()

    # Step 5: Scale annotated frame back to original dimensions for correct display
    h_orig, w_orig = frame_for_detection.shape[:2]
    if annotated.shape[:2] != (h_orig, w_orig):
        annotated = cv2.resize(annotated, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    boxes = results.boxes

    # If no objects detected, return the annotated frame with an empty list
    if len(boxes) == 0:
        return encode_frame(annotated, jpeg_quality=jpeg_quality, max_edge=preview_max_edge), [], annotated

    # Step 6: Extract detection data from GPU tensors to CPU NumPy arrays
    xyxy    = boxes.xyxy.cpu().numpy().astype(int)    # Bounding box corners [x1, y1, x2, y2]
    confs   = boxes.conf.cpu().numpy()                # Confidence scores 0.0–1.0
    cls_ids = boxes.cls.cpu().numpy().astype(int)     # Integer class IDs
    labels  = [model.names[i] for i in cls_ids]      # Human-readable class names

    # Build the structured detection list
    detections = [
        {
            "label":      str(labels[i]),
            "confidence": round(float(confs[i]), 2),
            "bbox":       [int(v) for v in xyxy[i]],  # Ensure plain Python ints for JSON
        }
        for i in range(len(labels))
    ]

    # Step 7: Encode annotated frame to Base64 JPEG
    return encode_frame(annotated, jpeg_quality=jpeg_quality, max_edge=preview_max_edge), detections, annotated


# ── WebSocket: Uploaded Video Processing ─────────────────────────────────────

@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    """
    Process an uploaded video file frame-by-frame with YOLO detection.

    Supports two client flows:
      Flow 1 (preferred): Client first POSTs to /upload-video, then sends:
                          {"temp_path": "uploads/<id>/original.mp4"} as a text message.
      Flow 2 (direct):    Client sends the entire video as a single binary WebSocket message.

    Uses a three-task async pipeline for concurrent read → detect → send:
      - reader():   Reads every UPLOAD_FRAME_SKIP-th frame from the video file.
      - detector(): Runs YOLO on each sampled frame in the background thread.
      - sender():   Writes annotated frames to processed.mp4 and streams results to browser.

    Sends per-frame JSON progress updates:
      {"frame": <base64>, "detections": [...], "processed_frames": int, "total_frames": int, ...}

    Sends a final completion message:
      {"done": true, "video_id": ..., "original_url": ..., "processed_url": ...}
    """
    await websocket.accept()
    model, _  = get_model()
    tmp_path  = None
    video_id  = None
    video_dir = None
    writer    = None
    cap       = None
    session_id = f"upload-{int(time.time() * 1000)}"  # Unique ID for this upload session

    try:
        # Wait for the first message to determine which upload flow is being used
        first_message = await websocket.receive()

        if first_message.get("text") is not None:
            # Flow 1: File was already saved by /upload-video — read the path from JSON
            msg_data = json.loads(first_message["text"])
            tmp_path = msg_data.get("temp_path")
            if not tmp_path or not os.path.exists(tmp_path):
                raise FileNotFoundError(f"Temporary video file not found at: {tmp_path}")
            video_dir = Path(tmp_path).parent
            video_id  = video_dir.name
        else:
            # Flow 2: Client sent the raw video bytes directly over WebSocket
            video_bytes = first_message.get("bytes")
            if not video_bytes:
                raise ValueError("Received empty video payload")
            video_id  = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            video_dir = UPLOAD_ROOT / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            tmp_path  = str(video_dir / "original.mp4")
            with open(tmp_path, "wb") as f:
                f.write(video_bytes)

        # Output path for the YOLO-annotated video
        processed_path = video_dir / "processed.mp4"

        # Open the video file and read its metadata
        cap                = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Unable to open uploaded video")

        fps                = cap.get(cv2.CAP_PROP_FPS) or 25
        source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # How many frames will actually be processed (every UPLOAD_FRAME_SKIP-th)
        total_frames = max(1, (source_total_frames + UPLOAD_FRAME_SKIP - 1) // UPLOAD_FRAME_SKIP)

        # Two async queues that connect the three pipeline tasks
        read_q   = asyncio.Queue(maxsize=8)  # reader  → detector
        result_q = asyncio.Queue(maxsize=8)  # detector → sender

        start_time = time.time()

        # Output video FPS matches the sampled rate so duration roughly matches the source
        writer_fps = max(1.0, fps / UPLOAD_FRAME_SKIP)

        log.info(
            "[%s] upload started: fps=%.2f source_frames=%s sampled_frames=%s skip=%s infer=%s conf=%.2f",
            session_id, fps, source_total_frames, total_frames,
            UPLOAD_FRAME_SKIP, UPLOAD_INFER_SIZE, UPLOAD_CONFIDENCE,
        )

        # ── Pipeline Task 1: Reader ───────────────────────────────────────────
        async def reader():
            """Read frames from the video file; put every UPLOAD_FRAME_SKIP-th frame into read_q."""
            idx  = 0
            loop = asyncio.get_running_loop()
            while True:
                # cap.read() is blocking — run in executor to keep event loop free
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    await read_q.put(None)  # Signal end-of-video to detector
                    break
                if idx % UPLOAD_FRAME_SKIP == 0:
                    await read_q.put((idx, frame.copy()))
                idx += 1

        # ── Pipeline Task 2: Detector ─────────────────────────────────────────
        async def detector():
            """Pull frames from read_q, run YOLO inference, push results to result_q."""
            loop      = asyncio.get_running_loop()
            processed = 0
            while True:
                item = await read_q.get()
                if item is None:
                    await result_q.put(None)  # Signal end to sender
                    break
                f_idx, frame  = item
                frame_started = time.time()

                # Run YOLO in background thread — non-blocking
                img_b64, detections, annotated = await loop.run_in_executor(
                    _executor,
                    run_detection,
                    model, frame,
                    UPLOAD_CONFIDENCE,
                    UPLOAD_INFER_SIZE,
                    UPLOAD_JPEG_QUALITY,
                    UPLOAD_PREVIEW_MAX_EDGE,
                )
                processed += 1
                infer_ms   = round((time.time() - frame_started) * 1000)
                await result_q.put((f_idx, processed, img_b64, detections, infer_ms, annotated))

        # ── Pipeline Task 3: Sender ───────────────────────────────────────────
        async def sender():
            """Pull results from result_q, write to video file, stream JSON to browser."""
            nonlocal writer
            while True:
                item = await result_q.get()
                if item is None:
                    break
                f_idx, processed, img_b64, detections, infer_ms, annotated = item

                # Initialize the video writer on the first annotated frame
                # (we need the frame dimensions to set up the writer)
                if writer is None:
                    h, w   = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(processed_path),
                        cv2.VideoWriter_fourcc(*"avc1"),
                        writer_fps,
                        (w, h),
                    )
                writer.write(annotated)

                # Calculate real-time throughput stats
                elapsed_ms     = round((time.time() - start_time) * 1000)
                processing_fps = round(processed / max(elapsed_ms / 1000, 0.001), 1)

                log.info(
                    "[%s] frame %s/%s source_idx=%s infer=%sms dets=%s stream_fps=%.2f",
                    session_id, processed, total_frames, f_idx,
                    infer_ms, len(detections), processing_fps,
                )

                # Stream frame data + progress metadata to the browser
                await websocket.send_text(json.dumps({
                    "frame":               img_b64,
                    "detections":          detections,
                    "frame_idx":           f_idx,
                    "processed_frames":    processed,
                    "total_frames":        total_frames,
                    "source_total_frames": source_total_frames,
                    "fps":                 round(fps, 1),
                    "processing_fps":      processing_fps,
                    "infer_ms":            infer_ms,
                    "elapsed_ms":          elapsed_ms,
                }))

        # Run all three pipeline tasks concurrently
        await asyncio.gather(reader(), detector(), sender())

        log.info("[%s] upload complete — stored at %s", session_id, video_dir)

        # Notify the browser that processing is done and provide download URLs
        await websocket.send_text(json.dumps({
            "done":          True,
            "video_id":      video_id,
            "original_url":  f"/videos/{video_id}/{Path(tmp_path).name}",
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
        # Always release file handles — even on disconnect or mid-stream errors.
        # Without this, processed.mp4 is written without its moov atom (unplayable/corrupt)
        # and the VideoCapture handle leaks.
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


# ── WebSocket: Browser Webcam Live Stream ─────────────────────────────────────

@app.websocket("/ws/stream")
async def stream_ws(websocket: WebSocket):
    """
    Accept a live webcam stream from a browser client.

    The browser captures frames via getUserMedia(), compresses them to JPEG,
    and sends each frame as a binary WebSocket message.
    The server runs YOLO on each frame and returns the annotated result.

    Client sends:  raw JPEG bytes (one frame per message)
    Server replies: {"frame": <base64>, "detections": [...], "timestamp": float}
    """
    await websocket.accept()
    model, _ = get_model()
    loop     = asyncio.get_running_loop()

    try:
        while True:
            # Receive raw JPEG bytes from the browser
            frame_bytes = await websocket.receive_bytes()

            # Decode JPEG bytes into an OpenCV BGR image array
            arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue  # Skip corrupted or empty frames

            # Run YOLO in background thread to avoid blocking the event loop
            img_b64, detections, _ = await loop.run_in_executor(
                _executor, run_detection, model, frame
            )

            # Return the annotated frame and detection results to the browser
            await websocket.send_text(json.dumps({
                "frame":      img_b64,
                "detections": detections,
                "timestamp":  time.time(),
            }))

    except WebSocketDisconnect:
        pass  # Browser tab closed — clean exit
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ── WebSocket: Camera Relay Ingest ────────────────────────────────────────────

@app.websocket("/ws/stream/{station_id}")
async def ingest_stream(websocket: WebSocket, station_id: str):
    """
    Receive JPEG frames streamed from local_relay.py running inside the restaurant.

    local_relay.py connects once per camera station and pushes raw JPEG binary
    frames continuously. This endpoint:
      1. Decodes each incoming JPEG frame.
      2. Runs YOLO detection in the background thread.
      3. Stores the latest detection snapshot in _snapshots (for Qubeyond verification).
      4. Fans out the annotated frame to all browser viewers watching this station.

    Client sends:  raw JPEG bytes (one frame per message)
    No response is sent back to the relay.
    """
    await websocket.accept()
    model, _ = get_model()
    loop     = asyncio.get_running_loop()

    print(f"[{station_id}] Relay connected from {websocket.client.host}")

    try:
        while True:
            # Receive compressed frame from the restaurant relay
            frame_bytes = await websocket.receive_bytes()

            # Decode JPEG bytes to OpenCV image
            arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue  # Skip corrupted frames

            # Run YOLO inference in background thread
            img_b64, detections, _ = await loop.run_in_executor(
                _executor, run_detection, model, frame
            )

            # Update the snapshot store — Qubeyond reads this during order verification
            _snapshots[station_id] = {
                "detections": detections,
                "frame_b64":  img_b64,
                "timestamp":  time.time(),
            }

            # Broadcast annotated frame to all browser viewers watching this station
            payload = json.dumps({
                "station_id": station_id,
                "frame":      img_b64,
                "detections": detections,
                "timestamp":  time.time(),
            })

            # Track disconnected viewers to clean them up after this loop iteration
            dead = set()
            for viewer in _viewers.get(station_id, set()):
                try:
                    await viewer.send_text(payload)
                except Exception:
                    dead.add(viewer)  # Mark as dead if send fails

            # Remove dead viewer connections from the registry
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


# ── WebSocket: Browser Viewer (live_monitor.html) ─────────────────────────────

@app.websocket("/ws/view/{station_id}")
async def view_stream(websocket: WebSocket, station_id: str):
    """
    Let a browser client subscribe to live annotated frames for a specific station.

    The relay pushes frames to /ws/stream/{station_id} → server processes them
    → results are broadcast here to all connected viewers.

    On connect, immediately sends the last known snapshot so the camera card
    is not blank while waiting for the next live frame.

    Sends a keepalive {"ping": true} every 30 seconds to prevent proxy timeouts.
    The browser tab can close at any time — the server cleans up automatically.
    """
    await websocket.accept()

    # Register this browser connection as a viewer for the requested station
    if station_id not in _viewers:
        _viewers[station_id] = set()
    _viewers[station_id].add(websocket)
    print(f"[{station_id}] Viewer connected ({len(_viewers[station_id])} total)")

    # Send the most recent cached frame immediately so the UI is not blank on load
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
        # Keepalive loop — ingest_stream pushes frames; this just keeps the connection alive
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        pass
    finally:
        # Remove this viewer from the registry when they disconnect
        _viewers.get(station_id, set()).discard(websocket)
        print(f"[{station_id}] Viewer disconnected")


# ── REST: Station Monitoring ───────────────────────────────────────────────────

@app.get("/ws/stations")
async def list_stations():
    """
    List all stations that currently have an active relay connection and viewer counts.

    Returns:
        relays:  List of station IDs that have sent at least one frame.
        viewers: Map of station_id → number of active browser viewers.
    """
    return {
        "relays":  list(_snapshots.keys()),
        "viewers": {sid: len(v) for sid, v in _viewers.items()},
    }


# ── REST: Health & Model Info ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Server health check endpoint.
    Returns server status and the compute device the YOLO model is running on.
    """
    _, device = get_model()
    return {"status": "ok", "device": device, "model": "YOLO"}


@app.get("/model-info")
async def model_info():
    """
    Return details about the loaded YOLO model.
    Includes the list of detectable food classes and inference settings.
    """
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
    """
    Return the latest detection snapshot for every active camera station.
    Used by Qubeyond integration and the frontend analytics dashboard.

    Returns per station:
        detections:  List of detected food items from the most recent frame.
        age_seconds: How many seconds ago the snapshot was captured.
    """
    return {
        sid: {
            "detections":  snap["detections"],
            "age_seconds": round(time.time() - snap["timestamp"], 1),
        }
        for sid, snap in _snapshots.items()
    }
