"""Service kích hoạt chấm lại (Task 15). Chạy từ src/:  python -m uvicorn api:app --port 8000
Cần MONGODB_URI và ML_API_TOKEN (Web gửi qua header X-ML-Token). Không có endpoint dự đoán từng khách."""
import hmac
import os
import threading

from fastapi import FastAPI, Header, HTTPException
from pymongo import MongoClient

import predict
import score_web

app = FastAPI(title="ML scoring")
_lock = threading.Lock()  # ponytail: khóa trong một tiến trình; chạy nhiều worker thì cần khóa ở DB


def _db():
    return MongoClient(os.environ["MONGODB_URI"]).get_default_database("test")  # URI không có tên db thì Express (mongoose) dùng "test"


@app.get("/health")
def health():
    return {"ok": True, "modelVersion": f"{predict.VERSION}-web"}


def _check_token(token: str | None) -> None:
    """Chưa đặt ML_API_TOKEN thì từ chối hết (fail closed); sai token thì 401."""
    expected = os.environ.get("ML_API_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Service ML chưa cấu hình ML_API_TOKEN")
    if not token or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(401, "Sai hoặc thiếu X-ML-Token")


@app.post("/score")
def score(x_ml_token: str | None = Header(default=None)):
    _check_token(x_ml_token)
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "Đang chấm điểm")
    try:
        return score_web.run(_db(), os.environ.get("ML_CUTOFF") or None)  # ML_CUTOFF: ghim mốc khi demo dữ liệu cũ
    finally:
        _lock.release()
