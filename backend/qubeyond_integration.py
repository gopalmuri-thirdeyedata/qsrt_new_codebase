"""
qubeyond_integration.py — KitchEye × Qubeyond POS Integration

This module handles all communication with the Qubeyond Point-of-Sale system.
It exposes a single webhook endpoint that Qubeyond calls when an ORDER_COMPLETE
event fires. The handler:

  1. Validates the HMAC-SHA256 request signature to reject forged requests.
  2. Deduplicates retried events (Qubeyond retries if it doesn't get a 200 response).
  3. Translates Qubeyond item names to KitchEye YOLO class names using ITEM_NAME_MAP.
  4. Reads the latest camera detection snapshot for the relevant kitchen station.
  5. Compares expected order items against detected items to find shortages.
  6. Returns the verification result (missing items, accuracy score, estimated savings).

Setup:
  1. Fill in QUBEYOND_API_KEY, QUBEYOND_WEBHOOK_SECRET etc. in backend/.env
  2. Deploy the server with a public URL (e.g. RunPod, Render, EC2)
  3. Register your webhook URL in the Qubeyond Portal:
       Settings → Integrations → Webhooks → Add Webhook
       URL:   https://your-server.com/webhook/qubeyond/kitchen-event
       Event: ORDER_COMPLETE

This file is already wired into main.py via:
  from qubeyond_integration import qubeyond_kitchen_event
  app.add_api_route("/webhook/qubeyond/kitchen-event", qubeyond_kitchen_event, methods=["POST"])
"""

# ── Standard Library Imports ──────────────────────────────────────────────────
import hmac                             # HMAC-based signature verification
import hashlib                          # SHA-256 hashing for HMAC computation
import asyncio                          # Async sleep for detection burst timing
from datetime import datetime           # UTC timestamps for order results
from typing import Optional             # Type hint: function may return None
from collections import defaultdict    # Dict that creates missing keys automatically

# ── Third-Party Imports ───────────────────────────────────────────────────────
from fastapi import Request, HTTPException  # Request body access and HTTP error responses
from dotenv import load_dotenv              # Load secrets from the .env file
import os                                   # Read environment variables

# ── Load Environment Configuration ───────────────────────────────────────────
# Reads all values from backend/.env into os.environ
load_dotenv()

# Qubeyond Configuration — all values come from backend/.env
QUBEYOND_WEBHOOK_SECRET = os.getenv("QUBEYOND_WEBHOOK_SECRET", "")  # HMAC signing secret
QUBEYOND_API_BASE_URL   = os.getenv("QUBEYOND_API_BASE_URL", "https://api.qubeyond.com")  # REST API root
QUBEYOND_API_KEY        = os.getenv("QUBEYOND_API_KEY", "")         # Bearer token for API calls
QUBEYOND_TENANT_ID      = os.getenv("QUBEYOND_TENANT_ID", "")       # Qubeyond account/tenant ID
QUBEYOND_OUTLET_ID      = os.getenv("QUBEYOND_OUTLET_ID", "")       # Restaurant location ID


# ── In-Memory Order Results Store ────────────────────────────────────────────
# Stores verification results for all processed orders in the current session.
# Key   = order_id (string)
# Value = full verification result dict
# NOTE: This is lost on server restart. Replace with SQLite or Postgres for production.
_order_results: dict = {}


# ── Menu Name Mapping ─────────────────────────────────────────────────────────
# Translates Qubeyond POS item names to KitchEye YOLO detection class labels.
#
# Keys   = YOLO class names (must exactly match the class names in best.pt)
# Values = List of Qubeyond item name strings that map to that class
#          (substring matching is used — case-insensitive)
#
# How to extend: Add new entries here when you add new food classes to your YOLO model.
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
    """
    Translate a Qubeyond item name to the matching KitchEye YOLO class label.

    Performs case-insensitive substring matching against ITEM_NAME_MAP.
    Returns None if the item is not in the menu map (it will be ignored).

    Example:
        _to_label("Large Fries")  → "fries"
        _to_label("Chicken Wrap") → "wrap"
        _to_label("Unknown Item") → None
    """
    name_lower = item_name.lower()
    for label, names in ITEM_NAME_MAP.items():
        if any(n.lower() in name_lower for n in names):
            return label
    return None  # Item not recognized by KitchEye — ignored in verification


# ── HMAC Signature Validation ─────────────────────────────────────────────────

async def _verify_signature(request: Request):
    """
    Validate the HMAC-SHA256 signature that Qubeyond attaches to every webhook request.

    This prevents attackers from forging fake order events.
    Qubeyond signs the raw request body using the shared QUBEYOND_WEBHOOK_SECRET.

    Validation is skipped if QUBEYOND_WEBHOOK_SECRET is blank (local dev convenience).
    In production, always configure the secret.

    Raises:
        HTTPException 401: If the signature header is missing or the hash does not match.
    """
    if not QUBEYOND_WEBHOOK_SECRET:
        return  # Skip validation in local dev when no secret is configured

    # Read the signature from the request header (strip the "sha256=" prefix if present)
    signature = (
        request.headers.get("x-qubeyond-signature") or
        request.headers.get("x-hub-signature-256", "")
    ).replace("sha256=", "")

    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature header")

    # Compute the expected HMAC-SHA256 hash of the raw request body
    body     = await request.body()
    expected = hmac.new(
        QUBEYOND_WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()

    # Use compare_digest for constant-time comparison (prevents timing attacks)
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")


# ── Event Deduplication ───────────────────────────────────────────────────────
# Qubeyond retries webhook delivery if it doesn't receive a 200 response within 5 seconds.
# This set tracks every processed event_id so duplicate deliveries are ignored.

_seen_events: set = set()


def _is_duplicate(event_id: str) -> bool:
    """
    Check whether this event_id has already been processed.

    Adds the event_id to the seen set if new.
    Trims the set to prevent unbounded memory growth (keeps latest 5,000 when over 10,000).

    Returns:
        True if the event was already processed (should be ignored).
        False if this is a new event (should be processed).
    """
    if event_id in _seen_events:
        return True

    _seen_events.add(event_id)

    # Prevent the set from growing indefinitely in long-running sessions
    if len(_seen_events) > 10_000:
        for e in list(_seen_events)[:5_000]:
            _seen_events.discard(e)

    return False


# ── Order Verification Logic ──────────────────────────────────────────────────

def _verify_order(expected: list, detected: list) -> dict:
    """
    Compare the items expected in a Qubeyond order against what the camera detected.

    For each expected item, counts how many were detected. Items with a shortage
    are added to the missing list. Calculates an accuracy score and estimated
    dollar savings from catching the error.

    Args:
        expected: List of dicts {"label": str, "quantity": int, "unit_price": float}
                  Built from the Qubeyond order payload using ITEM_NAME_MAP.
        detected: List of dicts {"label": str, "confidence": float, "bbox": [...]}
                  From the most recent YOLO snapshot for the station.

    Returns:
        {
            "missing":     [str, ...]       — e.g. ["fries ×1", "drink ×2"]
            "accuracy":    float            — 0.0 to 1.0; 1.0 = all items present
            "savings":     float            — estimated dollar value of missing items
            "intercepted": bool             — True if any items were missing
        }
    """
    if not expected:
        # Nothing expected → perfect by default (order has no trackable items)
        return {"missing": [], "accuracy": 1.0, "savings": 0.0, "intercepted": False}

    # Flatten detected labels into a simple list for counting
    detected_labels = [d.get("label", "").lower() for d in detected]

    missing = []
    savings = 0.0

    for item in expected:
        label    = item["label"]
        qty      = item["quantity"]
        price    = item.get("unit_price", 0.0)

        # Count how many of this label the camera actually detected
        found    = detected_labels.count(label)
        shortage = qty - found

        if shortage > 0:
            # Record shortfall and calculate dollar impact
            missing.append(f"{label} ×{shortage}")
            savings += shortage * price

    return {
        "missing":     missing,
        "accuracy":    round(1.0 - (len(missing) / len(expected)), 3),
        "savings":     round(savings, 2),
        "intercepted": len(missing) > 0,
    }


# ── Webhook Handler ───────────────────────────────────────────────────────────

async def qubeyond_kitchen_event(request: Request):
    """
    POST /webhook/qubeyond/kitchen-event

    The single endpoint Qubeyond calls when a kitchen order event occurs.
    Handles ORDER_COMPLETE and ORDER_BUMPED events.

    Processing flow:
      1. Validate HMAC signature → reject forged requests.
      2. Parse JSON body → extract event_type, order_id, station_id, items.
      3. Ignore non-order events (e.g. ORDER_CREATED).
      4. Deduplicate retries using event_id.
      5. Map Qubeyond item names → YOLO class labels using ITEM_NAME_MAP.
      6. Read camera detections for the station (from _snapshots in main.py).
      7. Compare expected vs detected → build verification result.
      8. Persist result in _order_results.
      9. Return result to Qubeyond within the 5-second timeout window.

    Returns:
        {"status": "verified", "order_id": ..., "missing": [...], "accuracy": float, "savings": float}
    """
    # Step 1: Validate request authenticity
    await _verify_signature(request)

    # Step 2: Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Extract key fields from the Qubeyond event payload
    event_type  = body.get("event_type", "")
    event_id    = body.get("event_id", "")
    order_id    = body.get("order_id", "")
    # station_id may be at the top level or nested inside metadata
    station_id  = body.get("station_id") or body.get("metadata", {}).get("station_id")
    outlet_name = body.get("outlet_name", "Unknown")
    items_raw   = body.get("items", [])

    # Step 3: Only process order completion events — ignore everything else
    if event_type not in ("ORDER_COMPLETE", "ORDER_BUMPED",
                          "order_complete", "order_bumped"):
        return {"status": "ignored", "reason": f"event_type '{event_type}' not handled"}

    # Step 4: Deduplicate — Qubeyond retries on network errors
    if event_id and _is_duplicate(event_id):
        return {"status": "duplicate"}

    # Step 5: Build expected item list from Qubeyond order payload
    # Group items by their YOLO label and sum quantities
    # Voided items (cancelled/refunded) are excluded
    counts = defaultdict(lambda: {"label": "", "quantity": 0, "unit_price": 0.0})
    for item in items_raw:
        if item.get("voided"):
            continue  # Skip cancelled items
        label = _to_label(item.get("item_name", ""))
        if label:
            counts[label]["label"]      = label
            counts[label]["quantity"]  += item.get("quantity", 1)
            counts[label]["unit_price"] = item.get("price", 0.0)
    expected = list(counts.values())

    if not expected:
        # Order contained no items that KitchEye can recognise — skip verification
        return {"status": "skipped", "reason": "no recognisable items in order"}

    # Step 6: Get camera detections for the relevant station.
    # Cloud-Relay Mode: reads the latest snapshot stored by ingest_stream() in main.py.
    # The snapshot is updated every frame by the local_relay → ingest_stream pipeline.
    import time

    if station_id:
        snapshots = getattr(request.app.state, "snapshots", {})
        snap      = snapshots.get(station_id)
        if snap:
            age = time.time() - snap.get("timestamp", 0)
            if age > 60:
                # Snapshot is stale — camera may be offline or relay disconnected
                print(f"[qubeyond] Warning: snapshot for {station_id} is {age:.0f}s old — camera may be offline")
            detected = snap.get("detections", [])
        else:
            # No snapshot yet — relay has not connected for this station
            print(f"[qubeyond] Warning: no snapshot found for station '{station_id}'")
            detected = []
    else:
        # Qubeyond did not include a station_id — cannot map order to a camera
        print("[qubeyond] Warning: event has no station_id — skipping camera verification")
        detected = []


    # Step 7: Compare expected order vs camera detections
    result = _verify_order(expected, detected)

    # Step 8: Persist the full verification record in memory
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

    # Step 9: Return the verification result to Qubeyond
    return {
        "status":   "verified",
        "order_id": order_id,
        **result,
    }