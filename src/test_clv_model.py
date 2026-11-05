"""Chạy: .venv/Scripts/python -m pytest src/test_clv_model.py  (từ gốc repo)"""
import numpy as np

from clv_model import inv_slog, slog, smear


def test_slog_roundtrip_keeps_sign():
    x = np.array([-5000.0, -1.0, 0.0, 0.5, 12345.0])
    assert np.allclose(inv_slog(slog(x)), x)
    assert np.array_equal(np.sign(slog(x)), np.sign(x))


def test_smearing_recovers_mean():
    # y = inv_slog(z + e): bỏ qua e thì đoán thấp có hệ thống; smearing phải khôi phục trung bình
    r = np.random.RandomState(0)
    z, e = r.uniform(5, 8, 20000), r.normal(0, 1.2, 20000)
    y = inv_slog(z + e)
    assert inv_slog(z).mean() / y.mean() < 0.6
    assert abs(smear(z, e[:2000]).mean() / y.mean() - 1) < 0.03


def test_order_value_by_category_and_fallback():
    import pandas as pd
    from clv_model import OrdersCLV, order_value_by_category
    tr = pd.DataFrame({"main_category": ["A", "A", "B"], "future_orders": [1, 3, 2], "future_profit_full_price": [100.0, 300.0, 1000.0]})
    v = order_value_by_category(tr)
    assert v["A"] == 100 and v["B"] == 500 and v["_all"] == 1400 / 6
    m = OrdersCLV("nb", None, [], v)
    assert list(m.order_value(pd.DataFrame({"main_category": ["B", "Z", None]}))) == [500, 1400 / 6, 1400 / 6]
