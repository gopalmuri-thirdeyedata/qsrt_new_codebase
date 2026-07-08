"""
qubeyond_webhook_test.py — Minimal Qubeyond webhook receiver for testing

Single endpoint: POST /webhook/qubeyond/kitchen-event

Run:
  pip install fastapi uvicorn
  uvicorn qubeyond_webhook_test:app --host 0.0.0.0 --port 8000 --reload

Test with curl:
  curl -X POST http://139.99.62.133:5040/webhook/qubeyond/kitchen-event \
    -H "Content-Type: application/json" \
    -d '{
      "event_type": "ORDER_COMPLETE",
      "event_id": "evt-001",
      "order_id": "ORD-1234",
      "order_number": "042",
      "outlet_name": "Vacaville",
      "station_id": "station_1",
      "items": [
        {"item_id": "1", "item_name": "Beef Burger", "quantity": 2, "price": 5.49},
        {"item_id": "2", "item_name": "Regular Fries", "quantity": 1, "price": 2.99},
        {"item_id": "3", "item_name": "Coca Cola",    "quantity": 2, "price": 1.99}
      ],
      "timestamp": "2026-05-19T14:30:00Z"
    }'
"""

import json
from datetime import datetime
from fastapi import FastAPI, Request

app = FastAPI(title="Qubeyond Webhook Test")


@app.post("/webhook/qubeyond/kitchen-event")
async def kitchen_event(request: Request):
    body = await request.json()

    # Print everything received so you can inspect the real payload shape
    print("\n" + "="*60)
    print(f"  Qubeyond event received: {datetime.utcnow().isoformat()}")
    print("="*60)
    print(json.dumps(body, indent=2))
    print("="*60 + "\n")

    # Qubeyond just needs a 200 back
    return {"status": "received", "order_id": body.get("order_id")}