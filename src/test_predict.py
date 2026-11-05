"""Chạy: .venv/Scripts/python -m pytest src/test_predict.py  (từ gốc repo, sau `python src/predict.py build`)"""
import pandas as pd
import pytest

import predict
from clean import clean_tables
from to_parquet import load_tables


@pytest.fixture(scope="module")
def scored():
    return predict.score(clean_tables(load_tables()), "2021-01-01")


def test_matches_notebook_validation(scored):
    # 52.771 khách ở mốc 2021-01 (Task 3); p_churn TB khớp p_cut lưu lúc build
    _, _, meta = predict.load()
    assert len(scored) == 52771
    assert scored.p_churn.mean() == pytest.approx(meta["p_cut"], abs=1e-9)


def test_columns_and_consistency(scored):
    assert {"p_churn", "clv", "segment", "net", "s_star", "top_reasons"} <= set(scored)
    assert scored.p_churn.between(0, 1).all() and (scored.clv > 0).all()
    assert len(scored.top_reasons.iloc[0]) == 3
    assert not any(r["feature"] == "cancel_rate" for rs in scored.top_reasons.head(500) for r in rs)
    # net > 0 <=> s_star < s_ref
    assert ((scored.net > 0) == (scored.s_star < predict.S_REF)).all()
    assert set(scored.segment) <= set(predict.SEGMENTS.values())


def test_send_voucher_rule(scored):
    # Quy tắc chính thức: ô Giữ ngay và net > 0; ma trận chọn ô, net chọn khách
    assert predict.SEGMENTS[(True, True)] == "Giữ ngay"
    expect = (scored.segment == "Giữ ngay") & (scored.net > 0)
    assert (scored.send_voucher == expect).all()
    assert (scored.clv[scored.send_voucher] > scored.voucher_cost[scored.send_voucher]).all()
    assert scored.send_voucher.any() and not scored.send_voucher.all()
