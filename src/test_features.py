"""Chạy: .venv/Scripts/python -m pytest src  (từ gốc repo)"""
import pandas as pd
import pytest

from clean import clean_tables
from config import TRAIN_CUTOFFS, VALID_CUTOFF
from features import FEATURES, build_snapshot
from to_parquet import load_tables


def truncate(raw: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Giả lập 'đứng ở ngày cutoff': xóa mọi bản ghi phát sinh từ cutoff trở đi."""
    t = dict(raw)
    t["orders"] = raw["orders"][raw["orders"].order_date < cutoff]
    kept = t["orders"].order_id
    t["order_items"] = raw["order_items"][raw["order_items"].order_id.isin(kept)]
    t["payments"] = raw["payments"][raw["payments"].order_id.isin(kept)]
    t["returns"] = raw["returns"][raw["returns"].return_date < cutoff]
    t["shipments"] = raw["shipments"][raw["shipments"].delivery_date < cutoff]
    t["reviews"] = raw["reviews"][raw["reviews"].review_date < cutoff]
    return t


@pytest.fixture(scope="module")
def raw():
    return load_tables()


@pytest.mark.parametrize("cutoff", [TRAIN_CUTOFFS[0], VALID_CUTOFF], ids=lambda d: f"{d:%Y-%m}")
def test_no_leakage(raw, cutoff):
    full = build_snapshot(clean_tables(raw), cutoff)[list(FEATURES)]
    past_only = build_snapshot(clean_tables(truncate(raw, cutoff)), cutoff)[list(FEATURES)]
    pd.testing.assert_frame_equal(full, past_only, check_categorical=False)


def _tiny_tables():
    d = pd.Timestamp
    orders = pd.DataFrame([
        # id, ngày, khách, trạng thái
        (1, d("2020-06-01"), "A", "delivered"), (2, d("2021-03-01"), "A", "returned"),
        (3, d("2020-06-01"), "B", "delivered"),
        (4, d("2017-06-01"), "C", "delivered"),
        (5, d("2020-06-01"), "D", "delivered"), (6, d("2021-02-01"), "D", "cancelled"),
        (7, d("2020-06-01"), "E", "delivered"), (8, d("2022-02-01"), "E", "delivered"),
        (9, d("2021-02-01"), "F", "delivered"),
    ], columns=["order_id", "order_date", "customer_id", "order_status"])
    orders = orders.assign(zip=1, payment_method="cod", device_type="mobile", order_source="direct")
    items = pd.DataFrame({"order_id": orders.order_id, "product_id": 1, "quantity": 2, "unit_price": 100.0,
                          "discount_amount": 0.0, "promo_id": None, "promo_id_2": None})
    return {
        "orders": orders,
        "order_items": items,
        "products": pd.DataFrame({"product_id": [1], "category": ["Casual"], "price": [100.0], "cogs": [60.0]}),
        "payments": pd.DataFrame({"order_id": orders.order_id, "payment_method": "cod", "payment_value": 200.0, "installments": 1}),
        # A trả 1 trong 2 sản phẩm của đơn tương lai, được hoàn 100
        "returns": pd.DataFrame({"return_id": ["R1"], "order_id": [2], "product_id": [1], "return_date": [d("2021-03-10")],
                                 "return_reason": ["wrong_size"], "return_quantity": [1], "refund_amount": [100.0]}),
        "reviews": pd.DataFrame({"review_id": ["V1"], "order_id": [1], "product_id": [1], "customer_id": ["A"],
                                 "review_date": [d("2020-06-10")], "rating": [5], "review_title": ["ok"]}),
        "shipments": pd.DataFrame({"order_id": [1, 3], "ship_date": [d("2020-06-02")] * 2,
                                   "delivery_date": [d("2020-06-05")] * 2, "shipping_fee": [1.0] * 2}),
        "customers": pd.DataFrame({"customer_id": list("ABCDEF"), "zip": 1, "city": "X", "signup_date": d("2015-01-01"),
                                   "gender": "Female", "age_group": "25-34", "acquisition_channel": "direct"}),
        "geography": pd.DataFrame({"zip": [1], "city": ["X"], "region": ["East"], "district": ["D1"]}),
    }


def test_labels_on_hand_made_data():
    s = build_snapshot(clean_tables(_tiny_tables()), "2021-01-01")
    # C: đơn cuối quá 3 năm trước mốc; F: chưa mua trước mốc -> không thuộc tập xét
    assert sorted(s.index) == ["A", "B", "D", "E"]
    # A mua lại trong cửa sổ; D chỉ có đơn hủy; E mua lại sau khi cửa sổ 12 tháng đã đóng
    assert s.churn.to_dict() == {"A": 0, "B": 1, "D": 1, "E": 1}
    # Lợi nhuận đơn 2 của A: doanh thu 200 − hoàn 100 − giá vốn 1 sản phẩm giữ lại 60 = 40
    assert s.loc["A", "future_gross_profit"] == pytest.approx(40.0)
    assert s.loc[["B", "D", "E"], "future_gross_profit"].eq(0).all()
    # Đơn hủy (D) và đơn ngoài cửa sổ (E) không tính vào số đơn tương lai
    assert s.future_orders.to_dict() == {"A": 1, "B": 0, "D": 0, "E": 0}
    assert s.loc["A", "future_profit_full_price"] == pytest.approx(40.0)  # không có giảm giá
    # Feature của A chỉ thấy đơn 1, không thấy lần trả hàng năm 2021
    assert s.loc["A", "frequency"] == 1
    assert s.loc["A", "recency_days"] == 214
    assert s.loc["A", "return_rate"] == 0
    assert s.loc["A", "monetary"] == pytest.approx(200.0)
    assert s.loc["D", "cancel_rate"] == 0  # đơn hủy nằm sau mốc
