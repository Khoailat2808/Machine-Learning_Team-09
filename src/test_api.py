"""Chạy: .venv/Scripts/python -m pytest src/test_api.py  (từ gốc repo). Không cần MongoDB: score_web.run được thay."""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_db", lambda: None)
    monkeypatch.setattr(api.score_web, "run", lambda db, cutoff: {"scored": 1, "scoredAt": None, "modelVersion": "v"})
    return TestClient(api.app)


def test_score_can_dung_token(client, monkeypatch):
    monkeypatch.setenv("ML_API_TOKEN", "bi-mat")
    assert client.post("/score").status_code == 401
    assert client.post("/score", headers={"X-ML-Token": "sai"}).status_code == 401
    ok = client.post("/score", headers={"X-ML-Token": "bi-mat"})
    assert ok.status_code == 200 and ok.json()["scored"] == 1


def test_score_chua_cau_hinh_token_thi_tu_choi(client, monkeypatch):
    monkeypatch.delenv("ML_API_TOKEN", raising=False)
    assert client.post("/score", headers={"X-ML-Token": "x"}).status_code == 503


def test_health_khong_can_token(client, monkeypatch):
    monkeypatch.setenv("ML_API_TOKEN", "bi-mat")
    assert client.get("/health").json()["ok"] is True
