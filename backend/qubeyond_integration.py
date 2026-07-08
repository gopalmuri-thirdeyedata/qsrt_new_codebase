"""
qubeyond_integration.py — Minimal Qubeyond kitchen events webhook

Only one endpoint: POST /webhook/qubeyond/kitchen-event
Qubeyond calls this when an order is complete.
We verify it against camera detections and store the result.

Setup:
  1. pip install python-dotenv
  2. Create .env with QUBEYOND_WEBHOOK_SECRET=your_secret
  3. Register https://your-server.com/webhook/qubeyond/kitchen-event
     in the Qubeyond portal under Settings → Integrations → Webhooks
  4. Select event: ORDER_COMPLETE

Wire into main.py — add these two lines:
  from qubeyond_integration import qubeyond_kitchen_event
  app.add_api_route(
      "/webhook/qubeyond/kitchen-event",
      qubeyond_kitchen_event,
      methods=["POST"]
  )
"""

import hmac
import hashlib
import asyncio
from datetime import datetime
from typing import Optional
from collections import defaultdict

from fastapi import Request, HTTPException
from dotenv import load_dotenv
import os

load_dotenv()
QUBEYOND_WEBHOOK_SECRET = os.getenv("QUBEYOND_WEBHOOK_SECRET", "")

# In-memory store of verification results
# Replace with a real DB (SQLite/Postgres) when you go to production
_order_results: dict = {}

# ── Menu mapping ──────────────────────────────────────────────────────────────
# Keys   = KitchEye YOLO class names (must match exactly)
# Values = Item name strings as they appear in Qubeyond order payload
# Extend this to cover your full menu

ITEM_NAME_MAP = {
    "burger":   ["Beef Burger", "Double Burger", "Cheeseburger"],
    "fries":    ["Regular Fries", "Large Fries", "Small Fries"],
    "drink":    ["Coca Cola", "Diet Coke", "Sprite", "Water"],
    "sandwich": ["Chicken Sandwich", "Fish Fillet"],
    "nuggets":  ["Chicken Nuggets", "Tenders"],
    "wrap":     ["Chicken Wrap", "Burrito"],
    "salad":    ["Garden Salad", "Caesar Salad"],
    "dessert":  ["Apple Pie", "Sundae", "Cookie"],
}

def _to_label(item_name: str) -> Optional[str]:
    """Map a Qubeyond item name to a KitchEye detection label."""
    name_lower = item_name.lower()
    for label, names in ITEM_NAME_MAP.items():
        if any(n.lower() in name_lower for n in names):
            return label
    return None  # item not in KitchEye menu — ignored


# ── Signature validation ──────────────────────────────────────────────────────

async def _verify_signature(request: Request):
    """
    Validate HMAC-SHA256 signature Qubeyond sends with every request.
    Prevents anyone else from calling your webhook.
    Verify the exact header name in the Qubeyond docs.
    """
    if not QUBEYOND_WEBHOOK_SECRET:
        return  # skip validation in local dev if secret not configured

    signature = (
        request.headers.get("x-qubeyond-signature") or
        request.headers.get("x-hub-signature-256", "")
    ).replace("sha256=", "")

    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature header")

    body     = await request.body()
    expected = hmac.new(
        QUBEYOND_WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")


# ── Deduplication ─────────────────────────────────────────────────────────────
# Qubeyond retries if it doesn't get a 200 — this prevents double-processing

_seen_events: set = set()

def _is_duplicate(event_id: str) -> bool:
    if event_id in _seen_events:
        return True
    _seen_events.add(event_id)
    # Trim to prevent unbounded growth
    if len(_seen_events) > 10_000:
        for e in list(_seen_events)[:5_000]:
            _seen_events.discard(e)
    return False


# ── Verification logic ────────────────────────────────────────────────────────

def _verify_order(expected: list, detected: list) -> dict:
    """
    Compare expected order items vs what the camera detected.
    Returns missing items, accuracy score, and dollar savings.
    """
    if not expected:
        return {"missing": [], "accuracy": 1.0,
                "savings": 0.0, "intercepted": False}

    detected_labels = [d.get("label", "").lower() for d in detected]
    missing = []
    savings = 0.0

    for item in expected:
        label    = item["label"]
        qty      = item["quantity"]
        price    = item.get("unit_price", 0.0)
        found    = detected_labels.count(label)
        shortage = qty - found
        if shortage > 0:
            missing.append(f"{label} ×{shortage}")
            savings += shortage * price

    return {
        "missing":     missing,
        "accuracy":    round(1.0 - (len(missing) / len(expected)), 3),
        "savings":     round(savings, 2),
        "intercepted": len(missing) > 0,
    }


# ── The one required endpoint ─────────────────────────────────────────────────

async def qubeyond_kitchen_event(request: Request):
    """
    Receives Qubeyond kitchen events via POST.
    This is the only endpoint Qubeyond needs.
    Must respond 200 within 5 seconds or Qubeyond will retry.
    """
    await _verify_signature(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type  = body.get("event_type", "")
    event_id    = body.get("event_id", "")
    order_id    = body.get("order_id", "")
    station_id  = body.get("station_id") or body.get("metadata", {}).get("station_id")
    outlet_name = body.get("outlet_name", "Unknown")
    items_raw   = body.get("items", [])

    # Only act on completed orders
    if event_type not in ("ORDER_COMPLETE", "ORDER_BUMPED",
                          "order_complete", "order_bumped"):
        return {"status": "ignored", "reason": f"event_type '{event_type}' not handled"}

    # Ignore Qubeyond retries for already-processed events
    if event_id and _is_duplicate(event_id):
        return {"status": "duplicate"}

    # Build expected item list — only items KitchEye knows about
    counts = defaultdict(lambda: {"label": "", "quantity": 0, "unit_price": 0.0})
    for item in items_raw:
        if item.get("voided"):
            continue
        label = _to_label(item.get("item_name", ""))
        if label:
            counts[label]["label"]      = label
            counts[label]["quantity"]  += item.get("quantity", 1)
            counts[label]["unit_price"] = item.get("price", 0.0)
    expected = list(counts.values())

    if not expected:
        return {"status": "skipped", "reason": "no recognisable items in order"}

    # Trigger camera detection on the relevant station, then read results
    from camera_pipeline import get_detections_for_station, force_detection_burst, _workers
    import time

    if station_id:
        if station_id in _workers:
            # Standalone / On-Premise Mode: Force camera worker burst
            force_detection_burst(station_id, duration=5.0)
            await asyncio.sleep(2.0)   # wait for camera to capture current tray state
            detected = get_detections_for_station(station_id)
        else:
            # Cloud-Relay Mode: Fallback to snapshots populated via websocket ingest in main.py
            snapshots = getattr(request.app.state, "snapshots", {})
            snap = snapshots.get(station_id)
            if snap:
                age = time.time() - snap.get("timestamp", 0)
                if age > 60:
                    print(f"[qubeyond] Warning: cloud snapshot for {station_id} is {age:.0f}s old")
                detected = snap.get("detections", [])
            else:
                detected = []
    else:
        detected = []

    # Compare expected vs detected
    result = _verify_order(expected, detected)

    # Persist result
    _order_results[order_id] = {
        "order_id":       order_id,
        "outlet_name":    outlet_name,
        "station_id":     station_id,
        "items_expected": expected,
        "items_detected": detected,
        **result,
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(
        f"[qubeyond] {outlet_name} | order={order_id} | "
        f"accuracy={result['accuracy']} | "
        f"missing={result['missing']} | "
        f"saved=${result['savings']}"
    )

    return {
        "status":   "verified",
        "order_id": order_id,
        **result,
    }