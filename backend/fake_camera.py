"""
fake_camera.py — Generates synthetic kitchen frames for local testing.
Creates a virtual RTSP-like stream using an HTTP MJPEG server.
local_relay.py reads from it just like a real IP camera.
Run this first, then run local_relay.py pointing at:
  http://localhost:8554/station_1
  http://localhost:8554/station_2
  etc.
Install:
  pip install opencv-python-headless numpy flask
Run:
  python fake_camera.py
"""



import cv2
import numpy as np
import time
import math
import threading
from flask import Flask, Response

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────
WIDTH  = 640
HEIGHT = 480
FPS    = 10

# Food items to render on the fake tray
FOOD_ITEMS = [
    {"name": "Burger",   "color": (50,  107, 255), "x": 0.25, "y": 0.50, "w": 0.18, "h": 0.22},
    {"name": "Fries",    "color": (0,   209, 255), "x": 0.50, "y": 0.50, "w": 0.14, "h": 0.20},
    {"name": "Drink",    "color": (239, 71,  111), "x": 0.72, "y": 0.50, "w": 0.10, "h": 0.24},
    {"name": "Nuggets",  "color": (111, 71,  239), "x": 0.38, "y": 0.50, "w": 0.16, "h": 0.16},
]

# ── Frame generator ───────────────────────────────────────────

def draw_frame(station_id: str, frame_count: int) -> np.ndarray:
    """Render a synthetic kitchen frame with food items on a tray."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Dark background
    frame[:] = (7, 10, 13)

    # Subtle grid
    for x in range(0, WIDTH, WIDTH // 10):
        cv2.line(frame, (x, 0), (x, HEIGHT), (20, 28, 38), 1)
    for y in range(0, HEIGHT, HEIGHT // 8):
        cv2.line(frame, (0, y), (WIDTH, y), (20, 28, 38), 1)

    # Tray surface
    tray_x1, tray_y1 = int(WIDTH * 0.08), int(HEIGHT * 0.25)
    tray_x2, tray_y2 = int(WIDTH * 0.92), int(HEIGHT * 0.80)
    cv2.rectangle(frame, (tray_x1, tray_y1), (tray_x2, tray_y2), (30, 44, 60), -1)
    cv2.rectangle(frame, (tray_x1, tray_y1), (tray_x2, tray_y2), (45, 65, 90), 2)

    # Simulate occasional missing item (every 5 seconds, remove one item)
    t = frame_count / FPS
    missing_idx = int(t / 5) % len(FOOD_ITEMS) if int(t) % 10 >= 5 else -1

    # Draw food items
    for i, item in enumerate(FOOD_ITEMS):
        if i == missing_idx:
            continue  # simulate missing item

        cx = int(item["x"] * WIDTH)
        cy = int(item["y"] * HEIGHT)
        bw = int(item["w"] * WIDTH)
        bh = int(item["h"] * HEIGHT)

        # Subtle wobble to simulate real camera
        jitter_x = int(math.sin(frame_count * 0.05 + i * 1.5) * 2)
        jitter_y = int(math.cos(frame_count * 0.04 + i * 2.0) * 1)
        cx += jitter_x
        cy += jitter_y

        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2

        # Item box (filled + border)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), item["color"], -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), item["color"], 2)

        # Item label
        label = item["name"]
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_x = cx - tw // 2
        label_y = y1 - 6
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    item["color"], 1, cv2.LINE_AA)

    # Station label top-left
    cv2.rectangle(frame, (6, 6), (180, 22), (0, 0, 0), -1)
    cv2.putText(frame, f"{station_id.upper()} — LIVE",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (90, 130, 160), 1, cv2.LINE_AA)

    # Timestamp bottom-right
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (WIDTH - 72, HEIGHT - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 90, 120), 1, cv2.LINE_AA)

    # Missing item alert
    if missing_idx >= 0:
        item_name = FOOD_ITEMS[missing_idx]["name"]
        alert_text = f"MISSING: {item_name}"
        (tw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ax = WIDTH // 2 - tw // 2
        cv2.rectangle(frame, (ax - 8, HEIGHT - 35), (ax + tw + 8, HEIGHT - 18),
                      (0, 0, 180), -1)
        cv2.putText(frame, alert_text, (ax, HEIGHT - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)

    return frame


# ── MJPEG stream generator ────────────────────────────────────

def generate_mjpeg(station_id: str):
    """Yield MJPEG frames for a given station."""
    frame_count = 0
    delay = 1.0 / FPS
    while True:
        t0 = time.monotonic()
        frame = draw_frame(station_id, frame_count)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpg = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        frame_count += 1
        elapsed = time.monotonic() - t0
        time.sleep(max(0, delay - elapsed))


# ── Flask routes ──────────────────────────────────────────────

@app.route("/station_<int:num>")
def stream(num: int):
    station_id = f"station_{num}"
    return Response(
        generate_mjpeg(station_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return """
    <h2>KitchEye Fake Camera Server</h2>
    <p>Available streams:</p>
    <ul>
      <li><a href="/station_1">Station 1</a> → http://localhost:8554/station_1</li>
      <li><a href="/station_2">Station 2</a> → http://localhost:8554/station_2</li>
      <li><a href="/station_3">Station 3</a> → http://localhost:8554/station_3</li>
      <li><a href="/station_4">Station 4</a> → http://localhost:8554/station_4</li>
    </ul>
    <p>Point local_relay.py at these URLs to simulate real cameras.</p>
    """

if __name__ == "__main__":
    print("[fake_camera] Synthetic camera server starting on http://localhost:8554")
    print("[fake_camera] Streams available:")
    for i in range(1, 5):
        print(f"  http://localhost:8554/station_{i}")
    print("[fake_camera] Food items rotate in/out every 5 seconds to simulate missing items")
    app.run(host="0.0.0.0", port=8554, threaded=True)