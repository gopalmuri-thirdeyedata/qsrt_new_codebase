# KitchEye — Automated Food Tray Audit System

> Real-time food order verification for QSR kitchens.  
> YOLOv8 object detection + Qubeyond POS integration + WebSocket live streaming.

---

## What It Does

KitchEye audits every food tray before it leaves the kitchen by:

1. **Watching** physical IP cameras over the restaurant LAN via a local relay agent.
2. **Detecting** food items on the tray in real-time using a custom-trained YOLOv8 model.
3. **Verifying** orders against Qubeyond POS data the moment an order is marked complete.
4. **Alerting** on missing or incorrect items — with an accuracy score and estimated savings.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Restaurant ["🏪 Restaurant (On-Site)"]
        CAM1["📷 IP Camera\nStation 1"]
        CAM2["📷 IP Camera\nStation 2"]
        RELAY["local_relay.py\n──────────────\nReads RTSP streams\nCompresses to JPEG\n5 fps per camera"]
        CAM1 -->|RTSP over LAN| RELAY
        CAM2 -->|RTSP over LAN| RELAY
    end

    subgraph Cloud ["☁️ Cloud Server (FastAPI)"]
        INGEST["WS /ws/stream/{station_id}\n──────────────────────\nReceives JPEG frames\nRuns YOLO detection\nStores latest snapshot"]
        SNAPSHOT["_snapshots store\n──────────────\nstation_id → detections\nTimestamp per frame"]
        VIEWER["WS /ws/view/{station_id}\n──────────────────────\nBroadcasts annotated\nframes to browser"]
        WEBHOOK["POST /webhook/qubeyond/kitchen-event\n────────────────────────────────────\nValidates HMAC signature\nTranslates item names\nCompares order vs camera\nReturns audit result"]
        UPLOAD["WS /ws/video\n──────────────\nUploads video file\nAnnotates every 5th frame\nSaves processed.mp4"]
        INGEST -->|"update"| SNAPSHOT
        INGEST -->|"fan-out"| VIEWER
        WEBHOOK -->|"read"| SNAPSHOT
    end

    subgraph Browser ["🖥️ Browser Dashboards"]
        DASH["demoDashboard.html\n──────────────────\nVideo upload mode\nWebcam live stream"]
        MON["live_monitor.html\n──────────────────\nMulti-station camera grid\nReal-time detections"]
        VIEWER -->|"annotated frames"| MON
        UPLOAD -->|"annotated frames"| DASH
    end

    subgraph POS ["💳 Qubeyond POS"]
        QB["Qubeyond Platform\n──────────────────\nORDER_COMPLETE event\nHMAC-signed webhook"]
        QB -->|"POST webhook"| WEBHOOK
        WEBHOOK -->|"audit result"| QB
    end

    RELAY -->|"JPEG binary\nWSS"| INGEST
```

---

## Project Structure

```
food-detection/
├── README.md
├── .gitignore
│
├── backend/
│   ├── main.py                   # FastAPI server — YOLO, WebSockets, REST API
│   ├── qubeyond_integration.py   # Qubeyond POS webhook + order verification
│   ├── local_relay.py            # Restaurant camera relay (runs on-site)
│   ├── requirements.txt          # Python dependencies
│   ├── model/                    # Subfolder containing all YOLO weights (.pt files)
│   │   ├── best.pt               # Custom trained YOLOv8 model
│   │   └── yolov8n.pt            # Fallback YOLOv8 nano model
│   ├── .env                      # Secrets — Qubeyond API keys, camera URLs
│   └── .env.example              # Safe template for teammates
│
└── frontend/
    ├── demoDashboard.html         # Operator dashboard — upload + webcam stream
    └── live_monitor.html          # Kitchen monitor — live multi-station camera grid
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A GPU is optional but recommended (CUDA or Apple Silicon MPS)

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env and fill in your values:
#   QUBEYOND_API_KEY=...
#   QUBEYOND_WEBHOOK_SECRET=...
#   CAM_1=rtsp://admin:password@192.168.1.101:554/stream1
```

### 3. Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open your browser at `http://localhost:8000`

### 4. Start the Camera Relay (inside the restaurant)

```bash
# Run on the local restaurant PC — connects cameras to the cloud server
python local_relay.py
```

---

## API Endpoints

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve operator dashboard (`demoDashboard.html`) |
| `GET` | `/monitor` | Serve live camera monitor (`live_monitor.html`) |
| `POST` | `/upload-video` | Upload a video file → returns `video_id` and `temp_path` |
| `GET` | `/health` | Server status and active compute device |
| `GET` | `/model-info` | YOLO class names and model settings |
| `GET` | `/stations` | Latest detections per camera station |
| `GET` | `/ws/stations` | Active relay connections and viewer counts |
| `POST` | `/webhook/qubeyond/kitchen-event` | Qubeyond POS order verification webhook |
| `GET` | `/videos/{video_id}/{filename}` | Download original or processed video |

### WebSocket

| Endpoint | Direction | Description |
|----------|-----------|-------------|
| `WS /ws/video` | ↕ | Upload video → receive annotated frames + progress |
| `WS /ws/stream` | ↕ | Push webcam JPEG frames → receive detections |
| `WS /ws/stream/{station_id}` | → | `local_relay.py` pushes camera frames to the server |
| `WS /ws/view/{station_id}` | ← | Browser subscribes to a live annotated station feed |

---

## Qubeyond Integration

### How It Works

1. An order is placed through the Qubeyond POS system.
2. The kitchen prepares the order — cameras watch the tray via `local_relay.py`.
3. When Qubeyond marks the order `ORDER_COMPLETE`, it sends a signed webhook to:
   ```
   POST /webhook/qubeyond/kitchen-event
   ```
4. KitchEye reads the most recent camera snapshot for that station.
5. Expected order items are compared against detected food items.
6. A verification result is returned to Qubeyond:

```json
{
  "status":      "verified",
  "order_id":    "ORD-1042",
  "missing":     ["fries ×1"],
  "accuracy":    0.833,
  "savings":     2.99,
  "intercepted": true
}
```

### Setup in Qubeyond Portal

1. Go to **Settings → Integrations → Webhooks → Add Webhook**
2. Set URL to: `https://your-server.com/webhook/qubeyond/kitchen-event`
3. Select event: `ORDER_COMPLETE`
4. Copy the generated **Webhook Secret** into your `.env` file:
   ```
   QUBEYOND_WEBHOOK_SECRET=your_secret_here
   ```

### Extending the Menu Map

Edit `ITEM_NAME_MAP` in `qubeyond_integration.py` to add new food items:

```python
ITEM_NAME_MAP = {
    "burger":   ["Beef Burger", "Double Burger", "Cheeseburger"],
    "fries":    ["Regular Fries", "Large Fries", "Small Fries"],
    # Add your new items here:
    "hotdog":   ["Hot Dog", "Chilli Dog"],
}
```

> Keys must exactly match your YOLO model's class names.

---

## Hardware & Compute

The server auto-detects the best available compute device at startup:

| Device | Requirement | Performance |
|--------|-------------|-------------|
| **CUDA** | NVIDIA GPU + PyTorch CUDA | Fastest (~10ms/frame) |
| **MPS** | Apple Silicon (M1/M2/M3) | Fast (~15ms/frame) |
| **CPU** | Any machine | Slower (~80–200ms/frame) |

FP16 (half precision) is automatically enabled on CUDA for ~2× speed.

---

## Camera Relay Configuration

Edit `.env` to configure cameras and the cloud server:

```env
# Cloud server address (where main.py is running)
CLOUD_WS_BASE=wss://your-server.com

# IP cameras in the restaurant (RTSP URLs)
CAM_1=rtsp://admin:password@192.168.1.101:554/stream1
CAM_2=rtsp://admin:password@192.168.1.102:554/stream1
CAM_3=rtsp://admin:password@192.168.1.103:554/stream1
CAM_4=rtsp://admin:password@192.168.1.104:554/stream1
```

Station IDs (`station_1`, `station_2`, ...) must match what Qubeyond sends
in the `station_id` field of order webhook events.

---

## Detection Classes

Your YOLO model (`best.pt`) determines the available food classes.  
Check them at runtime:

```bash
GET http://localhost:8000/model-info
```

Response:
```json
{
  "device":      "cuda",
  "classes":     {"0": "burger", "1": "fries", "2": "drink"},
  "num_classes": 3,
  "infer_size":  640
}
```
