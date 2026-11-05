"""Đóng gói và chấm điểm (Task 11).

Huấn luyện lại + lưu artifact (chạy từ gốc repo):   .venv/Scripts/python src/predict.py build
Chấm điểm:  score(tables, cutoff)  với `tables` = output của clean.clean_tables.

Mỗi khách trả về: p_churn (đã hiệu chỉnh), clv (CLV_giữ_được), segment, net (lợi nhuận kỳ vọng nếu phát voucher, tại
tỷ lệ cứu S_REF, dạng từng khách của công thức Neslin et al. 2006: p·s·(K − δ) − (1 − p)·δ), s_star (tỷ lệ cứu hòa vốn
(1 − p)δ / (p(K − δ)); vô cùng nếu K ≤ δ), send_voucher, top_reasons.
Kết luận Task 10: voucher % chỉ nên phát khi net > 0; xếp hạng không tính chi phí thì lỗ.
Quy tắc chính thức (tasks/voucher-rule-plan.md): ma trận chọn ô, net chọn khách → send_voucher = ô Giữ ngay và net > 0.
Nhãn phân khúc chỉ mô tả ô, không kèm hành động: hơn nửa ô Giữ ngay có net ≤ 0.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
MODELS = ML / "models"
VERSION = "1.0.0"
S_REF = 0.30  # tỷ lệ cứu cao nhất trong khoảng hợp lý (Task 10); chỉ để báo `net`, s_star không phụ thuộc
SEGMENTS = {(True, True): "Giữ ngay", (True, False): "Không chi tiền",
            (False, True): "Chăm sóc, không giảm giá", (False, False): "Để tự nhiên"}


# Bản web (Task 14): chỉ feature tính được từ users + orders của Web. Bỏ cancel_rate (đã chốt), review/return/giao hàng/trả góp/thiết bị/kênh/vùng (Web không có).
WEB_CHURN = ["recency_days", "frequency", "monetary", "mean_gap_days", "std_gap_days", "n_categories", "main_category"]
WEB_CLV = {"tenure_days", "avg_order_value", "mean_gap_days", "std_gap_days", "recency_ratio", "orders_3m", "orders_12m", "spend_trend",
           "active_years", "promo_share", "avg_discount_rate", "cod_share", "items_per_order", "n_categories"}
WEB_FEATURES = sorted(set(WEB_CHURN) | WEB_CLV)


def _suffix(web: bool) -> str:
    return "_web" if web else ""


def _prep(snap: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Đưa cột phân loại về đúng tập mức lúc train (mức lạ -> NaN)."""
    for c, cats in categories.items():
        snap[c] = pd.Categorical(snap[c].astype(str).where(snap[c].notna()), categories=cats)
    return snap


def _test_metrics(cm, clv, categories: dict) -> dict:
    """Metric trên test 2022-01 (bản web và bản chính cùng công thức, để so chênh)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from clean import clean_tables
    from config import TEST_CUTOFF
    from features import build_snapshot
    from to_parquet import load_tables
    te = _prep(build_snapshot(clean_tables(load_tables()), TEST_CUTOFF), categories)
    p, back = cm.predict_proba(te), te[te.churn == 0]
    return {"churn": {"roc_auc": roc_auc_score(te.churn, p), "pr_auc": average_precision_score(te.churn, p), "brier": brier_score_loss(te.churn, p)},
            "clv": {"spearman_orders": spearmanr(clv.expected_orders(back), back.future_orders)[0],
                    "spearman_clv": spearmanr(clv.predict(back), back.future_profit_full_price)[0],
                    "sum_pred_over_actual": clv.predict(back).sum() / back.future_profit_full_price.sum()}}


def build(web: bool = False) -> dict:
    """Fit lại hai model đóng băng, lưu churn.joblib, clv.joblib, metadata.json (web=True: churn_web.joblib, ... bản rút gọn cho Web)."""
    import churn_model, clv_model
    from churn_data import FINAL_FEATURES, load_churn_sets
    from config import L_YEARS, TEST_CUTOFF, TRAIN_CUTOFFS, VALID_CUTOFF, W_MONTHS
    from features import CATEGORICAL

    train, va = load_churn_sets()
    cm = churn_model.fit_final(WEB_CHURN if web else None)
    clv = clv_model.fit_final(WEB_CLV if web else None)
    oi = pd.read_parquet(ML / "data" / "interim" / "order_items.parquet")
    d = float((oi.discount_amount / (oi.quantity * oi.unit_price))[oi.promo_id.notna()].median())
    meta = {
        "version": VERSION, "W_months": W_MONTHS, "L_years": L_YEARS,
        "train_cutoffs": [str(c.date()) for c in TRAIN_CUTOFFS], "valid_cutoff": str(VALID_CUTOFF.date()), "test_cutoff": str(TEST_CUTOFF.date()),
        "churn_features": FINAL_FEATURES, "clv_features": clv.cols, "clv_scale": clv.scale,
        "categories": {c: [str(x) for x in train[c].cat.categories] for c in CATEGORICAL},
        "voucher_discount": d, "s_ref": S_REF,
        "p_cut": float(cm.predict_proba(va).mean()),  # ranh giới p_churn cao/thấp = p_churn TB validation
        "segment_rule": "p_churn cao: p >= p_cut; CLV cao: clv >= voucher_discount * avg_order_value",
        "variant": "web" if web else "main",
        "test_metrics": {"churn": json.loads((ML / "reports" / "churn_final.json").read_text(encoding="utf8"))["test"],
                         "clv": json.loads((ML / "reports" / "clv_final.json").read_text(encoding="utf8"))["test"]},
        "known_limits": ["cancel_rate nghi rò rỉ nhẹ (order_status là trạng thái cuối), không hiển thị làm lý do",
                         "tỷ lệ cứu voucher chưa đo được, net chỉ là kịch bản", "đơn vị tiền không xác định"],
    }
    if web:
        pr = pd.read_parquet(ML / "data" / "interim" / "products.parquet")
        ratio = (pr.cogs / pr.price).groupby(pr.category).median().to_dict()
        meta["cogs_ratio"] = {**ratio, "_all": float((pr.cogs / pr.price).median())}  # Web không có giá vốn
        meta["test_metrics"] = _test_metrics(cm, clv, meta["categories"])
        meta["known_limits"] = ["không có cancel_rate, review, trả hàng, giao hàng, trả góp, thiết bị, kênh, vùng", "giá vốn = tỷ suất trung vị theo danh mục",
                                "danh mục Web khác dataset -> main_category lạ rơi về mức trung bình", "tỷ lệ cứu voucher chưa đo được, net chỉ là kịch bản"]
    if clv.kind == "nb":
        clv.model.remove_data()  # bỏ dữ liệu train nhúng trong kết quả statsmodels (47 MB -> nhỏ)
    MODELS.mkdir(exist_ok=True)
    sfx = _suffix(web)
    joblib.dump(cm, MODELS / f"churn{sfx}.joblib")
    joblib.dump(clv, MODELS / f"clv{sfx}.joblib")
    (MODELS / f"metadata{sfx}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf8")
    return meta


def load(web: bool = False):
    sfx = _suffix(web)
    meta = json.loads((MODELS / f"metadata{sfx}.json").read_text(encoding="utf8"))
    return joblib.load(MODELS / f"churn{sfx}.joblib"), joblib.load(MODELS / f"clv{sfx}.joblib"), meta


def score(tables: dict[str, pd.DataFrame], cutoff, k_reasons: int = 3, web: bool = False) -> pd.DataFrame:
    """Một dòng mỗi khách có đơn hợp lệ trong L năm trước `cutoff`. Không cần nhãn tương lai."""
    from features import build_snapshot
    cm, clv, meta = load(web)
    snap = _prep(build_snapshot(tables, cutoff), meta["categories"])
    p, k = cm.predict_proba(snap), clv.predict(snap)
    cost = meta["voucher_discount"] * snap.avg_order_value.values
    out = pd.DataFrame({"customer_id": snap.index, "p_churn": p, "clv": k, "voucher_cost": cost}, index=snap.index)
    high_p = p >= meta["p_cut"]
    out["segment"] = [SEGMENTS[(a, b)] for a, b in zip(high_p, k >= cost)]
    # Neslin et al. (2006), từng khách, c = 0, A = 0: chi phí voucher chỉ phát sinh khi khách mua
    out["net"] = p * meta["s_ref"] * (k - cost) - (1 - p) * cost
    out["send_voucher"] = high_p & (k >= cost) & (out["net"].values > 0)
    out["s_star"] = np.where(k > cost, (1 - p) * cost / (p * (k - cost)), np.inf)
    out["top_reasons"] = cm.top_reasons(snap, k_reasons)
    out["cutoff"] = pd.Timestamp(cutoff)
    out["model_version"] = meta["version"]
    return out.reset_index(drop=True)


if __name__ == "__main__":
    if sys.argv[1:] in (["build"], ["build", "web"]):
        print(json.dumps({k: v for k, v in build(sys.argv[2:] == ["web"]).items() if k in ("version", "p_cut", "voucher_discount", "clv_scale", "test_metrics")}, default=float))
    else:
        print(__doc__)
