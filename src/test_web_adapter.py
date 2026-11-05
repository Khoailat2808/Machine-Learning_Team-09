"""Chạy: .venv/Scripts/python -m pytest src/test_web_adapter.py  (từ gốc repo). Dùng mongomock, không cần MongoDB thật.

Seed 80 khách DATATHON vào MongoDB giả theo schema Web, đọc lại bằng adapter, so feature với tính thẳng từ CSV.
`predict.score(web=True)` cần `python src/predict.py build web` đã chạy.
"""
import mongomock
import numpy as np
import pandas as pd
import pytest

import predict
import seed_demo
import web_adapter
from clean import clean_tables
from features import build_snapshot
from to_parquet import load_tables

CUTOFF = "2022-01-01"
COGS = {"_all": 0.5}


@pytest.fixture(scope="module")
def world():
    raw = load_tables()
    db = mongomock.MongoClient().db
    db.roles.insert_one({"name": "CUSTOMER"})
    counts = seed_demo.seed(db, raw, 80)
    return raw, db, counts


def test_seed_only_tagged_and_purge(world):
    _, db, counts = world
    assert counts["users"] == 80 and counts["orders"] > 80
    assert all(d["demoTag"] == "datathon" for c in seed_demo.COLLECTIONS for d in db[c].find())
    db.users.insert_one({"name": "thật"})  # purge không được đụng bản ghi không có cờ
    other = mongomock.MongoClient().db2
    other.roles.insert_one({"name": "CUSTOMER"})
    seed_demo.seed(other, world[0], 5)
    other.users.insert_one({"name": "thật"})
    assert seed_demo.purge(other)["users"] == 5 and other.users.count_documents({}) == 1


def test_features_match_csv(world):
    raw, db, _ = world
    ref = {str(u["_id"]): u["demoRef"] for u in db.users.find({"demoTag": "datathon"})}
    web = build_snapshot(web_adapter.load_tables(db, COGS), CUTOFF)
    web.index = web.index.map(ref)
    # Web chưa có dữ liệu trả hàng: `monetary` của Web = doanh thu chưa trừ tiền hoàn, nên đối chiếu với CSV đã bỏ bảng returns
    csv = build_snapshot(clean_tables({**raw, "returns": raw["returns"].iloc[0:0]}), CUTOFF)
    csv = csv.loc[csv.index.isin(ref.values())]

    assert len(web) > 20 and set(web.index) == set(csv.index)
    csv = csv.loc[web.index]
    num = [f for f in predict.WEB_FEATURES if f != "main_category"]
    pd.testing.assert_frame_equal(web[num].astype(float), csv[num].astype(float), check_exact=False, rtol=1e-9, atol=1e-9)
    assert (web.main_category.astype(str) == csv.main_category.astype(str)).all()
    assert (web.churn == csv.churn).all() and (web.future_orders == csv.future_orders).all()


def test_score_web_runs(world):
    _, db, _ = world
    s = predict.score(web_adapter.load_tables(db), CUTOFF, web=True)
    assert len(s) > 20 and s.p_churn.between(0, 1).all() and np.isfinite(s.clv).all()
    assert not any(r["feature"] == "cancel_rate" for rs in s.top_reasons for r in rs)


def test_score_web_contract_and_api(world, tmp_path, monkeypatch):
    import api
    import export_demo
    from fastapi.testclient import TestClient
    _, db, _ = world
    monkeypatch.setattr(api, "_db", lambda: db)
    monkeypatch.setenv("ML_API_TOKEN", "t")
    c = TestClient(api.app, headers={"X-ML-Token": "t"})
    assert c.get("/health").json()["ok"]
    from bson import ObjectId
    db.customer_scores.insert_one({"customerId": ObjectId(), "segment": "LET_BE"})  # điểm của khách không còn được chấm
    r = c.post("/score").json()
    n = r["scored"]
    assert n > 0 and db.customer_scores.count_documents({}) == n
    d = db.customer_scores.find_one()
    assert set(d) >= {"customerId", "pChurn", "clv", "segment", "topReasons", "net", "sStar", "voucherCost", "isControl", "scoredAt", "cutoff", "modelVersion", "schemaVersion"}
    assert d["segment"] in {"RETAIN", "NURTURE", "NO_SPEND", "LET_BE"} and len(d["topReasons"]) <= 3
    ctrl = {x["customerId"]: x["isControl"] for x in db.customer_scores.find()}
    assert 0 < sum(ctrl.values()) < n
    c.post("/score")  # chấm lại: upsert không nhân đôi, nhóm đối chứng giữ nguyên
    assert db.customer_scores.count_documents({}) == n
    assert ctrl == {x["customerId"]: x["isControl"] for x in db.customer_scores.find()}
    assert api._lock.acquire() and c.post("/score").status_code == 409  # đang chạy -> 409
    api._lock.release()
    assert export_demo.export(db, tmp_path)["customer_scores"] == n
