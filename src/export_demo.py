"""Xuất users.json, orders.json, customer_scores.json (bản ghi demoTag='datathon') cho Web seed khi không có ML (Task 15).

    .venv/Scripts/python src/export_demo.py [thư_mục_đích]      # mặc định export/
customer_scores chỉ gồm khách demo. Định dạng MongoDB Extended JSON ({"$oid"}, {"$date"}).
"""
import os
import sys
from pathlib import Path

from bson import json_util
from pymongo import MongoClient


def export(db, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    demo_ids = [u["_id"] for u in db.users.find({"demoTag": "datathon"}, {"_id": 1})]
    parts = {"users": db.users.find({"demoTag": "datathon"}), "orders": db.orders.find({"demoTag": "datathon"}),
             "customer_scores": db.customer_scores.find({"customerId": {"$in": demo_ids}})}
    counts = {}
    for name, cur in parts.items():
        rows = list(cur)
        (out / f"{name}.json").write_text(json_util.dumps(rows, ensure_ascii=False, indent=1), encoding="utf8")
        counts[name] = len(rows)
    return counts


if __name__ == "__main__":
    uri = os.environ.get("MONGODB_URI") or sys.exit("Thiếu biến môi trường MONGODB_URI")
    print(export(MongoClient(uri).get_default_database(), Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "export")))
