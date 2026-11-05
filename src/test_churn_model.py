"""Chạy: .venv/Scripts/python -m pytest src/test_churn_model.py  (từ gốc repo)"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from churn_data import load_churn_sets
from churn_model import fit_final


@pytest.fixture(scope="module")
def setup():
    _, va = load_churn_sets()
    return fit_final(), va


def test_reproduces_notebook_score(setup):
    m, va = setup
    assert average_precision_score(va.churn, m.predict_proba(va)) == pytest.approx(0.8595, abs=2e-4)


def test_contributions_sum_to_raw_logit(setup):
    m, va = setup
    x = va.head(200)
    raw = m.model.predict_proba(x[m.features])[:, 1]
    assert np.allclose(m.contributions(x).sum(axis=1), np.log(raw / (1 - raw)), atol=1e-6)


def test_top_reasons_shape(setup):
    m, va = setup
    r = m.top_reasons(va.head(5))
    assert len(r) == 5 and all(len(x) == 3 for x in r)
    assert all(abs(x[0]["contribution"]) >= abs(x[2]["contribution"]) for x in r)


def test_top_reasons_hide_cancel_rate(setup):
    m, va = setup
    x = va[va.cancel_rate > 0].head(500)  # nhóm mà cancel_rate có đóng góp khác 0
    assert all(r["feature"] != "cancel_rate" for rs in m.top_reasons(x) for r in rs)
