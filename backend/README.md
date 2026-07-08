# Food Detection Backend

## Setup

```bash
pip install -r requirements.txt
```

Place your trained YOLO model as `yolo_food.pt` in this directory.  
If the file is missing, the server falls back to `yolov8n.pt` (auto-downloaded).

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `GET /health` | HTTP | Server + device status |
| `GET /model-info` | HTTP | Loaded model classes |
| `WS /ws/video` | WebSocket | Upload video → streamed frames + detections |
| `WS /ws/stream` | WebSocket | Send JPEG frames → detections per frame |

## WebSocket Protocols

### `/ws/video`
1. Client connects
2. Client sends video file as a **single binary message**
3. Server streams JSON frames:
```json
{ "frame": "<base64-jpeg>", "detections": [...], "frame_idx": 0, "total_frames": 300, "fps": 25 }
```
4. Server sends `{ "done": true }` when complete

### `/ws/stream`
1. Client connects
2. Client sends JPEG frames as **binary messages** (one per frame)
3. Server responds per frame:
```json
{ "frame": "<base64-jpeg>", "detections": [...], "timestamp": 1234567890.0 }
```
