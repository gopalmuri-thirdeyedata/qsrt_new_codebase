# KitchEye — Food Detection System

A full-stack food item detection system for fast food kitchen quality control.  
Uses YOLO (Ultralytics) for real-time object detection with GPU acceleration (CUDA / MPS).

## Architecture

```
kitcheye/
├── backend/          FastAPI + YOLO + WebSockets
└── frontend/         React + Recharts UI
```

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

# Place your trained model here (or yolov8n.pt will be used as fallback):
cp /path/to/your/yolo_food.pt .

uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm start         # Dev server on :3000
# or
npm run build     # Production build
```

### 3. Environment (optional)

Create `frontend/.env`:
```
REACT_APP_WS_URL=ws://localhost:8000
```

---

## How It Works

### Video Upload Mode
1. Operator selects a pre-recorded `.mp4` / `.mov` file
2. File is sent as a single binary WebSocket message to `/ws/video`
3. Backend processes frame-by-frame using YOLO on GPU
4. Annotated frames + detection JSON streamed back in real-time
5. Progress bar shows % complete

### Live Stream Mode
1. Browser accesses webcam via `getUserMedia`
2. Canvas captures frames at 10 fps → sent as JPEG blobs to `/ws/stream`
3. Backend runs YOLO on each frame (async, non-blocking)
4. Annotated frame + detections returned per message

### GPU Selection
Backend auto-selects: **CUDA → MPS (Apple Silicon) → CPU**

---

## UI Views

### Operator Console
- Mode toggle: Video Upload / Live Stream
- Real-time annotated video feed
- Live detection panel with confidence bars
- Order queue with accuracy scoring (green/yellow/red)

### Analytics Dashboard (Owner)
- 6 summary KPI cards
- Accuracy trend chart (weekly / monthly)
- Order volume bar chart
- Per-item accuracy horizontal bar
- Error breakdown by item

---

## Adding POS Integration

When your POS system is ready:

1. Expose an endpoint (or message queue) that emits order payloads:
```json
{ "order_id": "ORD-1001", "items": { "Burger": 2, "Fries": 1 } }
```

2. In `backend/main.py`, compare detected items against the order:
```python
def verify_order(expected: dict, detected: list) -> dict:
    counts = {}
    for d in detected:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    accuracy = sum(1 for k,v in expected.items() if counts.get(k,0)==v) / len(expected)
    return {"accuracy": accuracy, "matched": counts}
```

3. Include verification results in the WebSocket payload.

---

## Detection Classes

Your YOLO model determines available classes. Check them at runtime:
```
GET http://localhost:8000/model-info
```
