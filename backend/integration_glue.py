"""
integration_glue.py — Wire camera_pipeline.py into main.py and qubeyond_integration.py

This file shows the exact changes needed to connect everything together.
Don't run this file directly — copy the relevant sections into your existing files.
"""

# ════════════════════════════════════════════════════════════════
# CHANGE 1: main.py — add to the top imports section
# ════════════════════════════════════════════════════════════════

MAIN_PY_IMPORTS = """
# Add these imports at the top of main.py
from camera_pipeline import start_all_cameras, get_all_station_states
from camera_pipeline import app as pipeline_app   # import routes
"""

# ════════════════════════════════════════════════════════════════
# CHANGE 2: main.py — update the startup event
# ════════════════════════════════════════════════════════════════

MAIN_PY_STARTUP = """
# Replace your existing startup_event in main.py with this:
@app.on_event("startup")
async def startup_event():
    # 1. Load YOLO model (existing code)
    global _model, _device
    loop = asyncio.get_running_loop()
    _model, _device = await loop.run_in_executor(_executor, load_model)
    print("[kitcheye] Model ready")

    # 2. Start all camera workers (new)
    await start_all_cameras()
    print("[kitcheye] All cameras started")
"""

# ════════════════════════════════════════════════════════════════
# CHANGE 3: qubeyond_integration.py — replace the stub function
# ════════════════════════════════════════════════════════════════

QUBEYOND_STUB_REPLACEMENT = """
# In qubeyond_integration.py, replace:
#
#   def get_latest_detections_for_order(order_id: str) -> list[dict]:
#       return []
#
# With this:

from camera_pipeline import get_detections_for_station, force_detection_burst

def get_latest_detections_for_order(order_id: str, station_id: str = None) -> list[dict]:
    \"\"\"
    Get latest camera detections for order verification.
    station_id comes from the Qubeyond event payload.
    \"\"\"
    if not station_id:
        return []
    return get_detections_for_station(station_id)
"""

# ════════════════════════════════════════════════════════════════
# CHANGE 4: qubeyond_integration.py — update webhook handler
# to force detection AND pass station_id to verification
# ════════════════════════════════════════════════════════════════

QUBEYOND_WEBHOOK_UPDATE = """
# In qubeyond_integration.py webhook handler, update this section:

    # BEFORE (original):
    # detected = get_latest_detections_for_order(event.order_id)

    # AFTER — pass station_id and trigger detection burst:
    station_id = event.station_id or event.metadata.get("station_id")

    # Force a detection burst on this station so we capture
    # the final tray state right now (handles the "too late?" problem)
    if station_id:
        from camera_pipeline import force_detection_burst
        force_detection_burst(station_id, duration=5.0)
        # Give cameras 2 seconds to capture the current state
        await asyncio.sleep(2.0)

    detected = get_latest_detections_for_order(event.order_id, station_id)
"""

# ════════════════════════════════════════════════════════════════
# CHANGE 5: camera_pipeline.py — update CAMERAS dict
# ════════════════════════════════════════════════════════════════

CAMERA_CONFIG_EXAMPLE = """
# In camera_pipeline.py, update CAMERAS to match your actual streams:

CAMERAS = {
    # Station IDs must match what Qubeyond sends in station_id field
    # Get your actual RTSP URLs from your camera admin panels

    "station_1": "rtsp://admin:password@192.168.1.101:554/stream1",
    "station_2": "rtsp://admin:password@192.168.1.102:554/stream1",
    "station_3": "rtsp://admin:password@192.168.1.103:554/stream1",
    "station_4": "rtsp://admin:password@192.168.1.104:554/stream1",

    # HTTP MJPEG streams (some IP cameras use this instead of RTSP):
    # "station_1": "http://192.168.1.101/video/mjpg.cgi",

    # Local USB webcam (for testing):
    # "station_test": 0,
}

# Also tune motion sensitivity for your kitchen environment:
MOTION_THRESHOLD = 3000   # increase if getting too many false triggers
                           # decrease if missing real motion
COOLDOWN_SECONDS = 5.0    # increase for slower packing workflows
"""

# ════════════════════════════════════════════════════════════════
# CHANGE 6: Frontend operator view — connect to multi-camera WS
# ════════════════════════════════════════════════════════════════

FRONTEND_MULTICAM = """
// In your React OperatorView or a new MultiCamView component:
// Connect to /ws/all-stations to receive all camera feeds

function useAllStations() {
  const [stations, setStations] = useState({});

  useEffect(() => {
    const ws = new WebSocket('ws://your-server/ws/all-stations');
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.ping) return;
      setStations(prev => ({
        ...prev,
        [msg.station_id]: msg   // keyed by station
      }));
    };
    return () => ws.close();
  }, []);

  return stations;
}

// Render 4 camera feeds in a grid:
function MultiCamView() {
  const stations = useAllStations();

  return (
    <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'8px'}}>
      {Object.entries(stations).map(([id, data]) => (
        <div key={id}>
          <div>{id} — {data.state}</div>
          {data.frame && <img src={`data:image/jpeg;base64,${data.frame}`} />}
          <div>{data.detections?.map(d => d.label).join(', ')}</div>
        </div>
      ))}
    </div>
  );
}
"""

print("Integration guide loaded — see comments above for each change needed.")
print("\nFile summary:")
print("  camera_pipeline.py       — multi-camera motion detection + YOLO")
print("  qubeyond_integration.py  — webhook receiver + order verification")
print("  integration_glue.py      — this file, shows how to wire them together")
print("\nChanges needed in existing files:")
print("  main.py                  — add start_all_cameras() to startup event")
print("  qubeyond_integration.py  — replace stub function, add force_detection_burst()")