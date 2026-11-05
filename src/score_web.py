"""Chấm điểm khách của Web và upsert vào `customer_scores` (Task 15). Cần MONGODB_URI.

    .venv/Scripts/python src/score_web.py [--cutoff 2023-01-01]   # mặc định: ngày sau đơn mới nhất
Hợp đồng dữ liệu: README.md (mục "Hợp đồng ML ↔ Web").
"""
import argparse
import hashlib
import math
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from bson import ObjectId

import predict
import web_adapter

SCHEMA_VERSION = 1
CONTROL_PCT = 10
# Khóa theo (p cao, CLV cao) của predict.SEGMENTS để đổi nhãn tiếng Việt không làm vỡ mã gửi sang Web
SEGMENT_CODE = {predict.SEGMENTS[k]: code for k, code in {(True, True): "RETAIN", (False, True): "NURTURE",
                                                          (True, False): "NO_SPEND", (False, False): "LET_BE"}.items()}


def is_control(customer_id) -> bool:
    """Hash của customerId, không phụ thuộc phân khúc/lần chấm: nhóm đối chứng giữ nguyên, ~10% mỗi phân khúc."""
    return int(hashlib.md5(str(customer_id).encode()).hexdigest(), 16) % 100 < CONTROL_PCT


def _num(x):  # s_star vô cùng (clv ≤ chi phí voucher) -> None, để JSON/Mongo không giữ inf
    return None if not math.isfinite(x) else float(x)


def to_docs(scores: pd.DataFrame, scored_at: datetime, version: str) -> list[dict]:
    return [{"customerId": ObjectId(r.customer_id), "pChurn": float(r.p_churn), "clv": float(r.clv),
             "segment": SEGMENT_CODE[r.segment],
             "topReasons": [{"feature": x["feature"], "direction": "UP" if x["effect"] == "tăng rủi ro" else "DOWN"} for x in r.top_reasons],
             "net": float(r.net), "sStar": _num(r.s_star), "voucherCost": float(r.voucher_cost),
             "isControl": is_control(r.customer_id), "scoredAt": scored_at, "cutoff": r.cutoff.to_pydatetime(),
             "modelVersion": version, "schemaVersion": SCHEMA_VERSION} for r in scores.itertuples()]


def run(db, cutoff=None) -> dict:
    t = web_adapter.load_tables(db)
    if t["orders"].empty:
        return {"scored": 0}
    cutoff = pd.Timestamp(cutoff) if cutoff else t["orders"].order_date.max().normalize() + timedelta(days=1)
    scores = predict.score(t, cutoff, web=True)
    docs = to_docs(scores, datetime.utcnow(), f"{predict.VERSION}-web")
    if docs:
        db.customer_scores.create_index("customerId", unique=True)
        for d in docs:  # ponytail: từng lệnh một (mongomock chưa hỗ trợ bulk_write của pymongo 4.19); hàng chục nghìn khách thì đổi sang bulk_write
            db.customer_scores.replace_one({"customerId": d["customerId"]}, d, upsert=True)
        # Khách không còn được chấm (đã xóa/không còn hoạt động) thì bỏ điểm cũ, kẻo vẫn tính vào tổng CLV bị đe dọa.
        db.customer_scores.delete_many({"customerId": {"$nin": [d["customerId"] for d in docs]}})
    return {"scored": len(docs), "scoredAt": docs[0]["scoredAt"] if docs else None, "modelVersion": f"{predict.VERSION}-web"}


if __name__ == "__main__":
    from pymongo import MongoClient
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff")
    uri = os.environ.get("MONGODB_URI") or sys.exit("Thiếu biến môi trường MONGODB_URI")
    print(run(MongoClient(uri).get_default_database("test"), ap.parse_args().cutoff))  # URI không có tên db thì Express (mongoose) dùng "test"
