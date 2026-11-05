"""Phần 2 của mô hình CLV hai phần: lợi nhuận 12 tháng tới NẾU khách quay lại (`CLV_giữ_được`).

Phần 1 là model churn (`churn_model.py`). `CLV_kỳ_vọng = (1 − p_churn) × CLV_giữ_được`.

Model chính (`OrdersCLV`, hurdle Negative Binomial): CLV_giữ_được = E[số đơn | quay lại] × lợi nhuận giá gốc TB mỗi đơn theo danh mục × scale.
Số đơn thêm (future_orders − 1 ≥ 0) dự báo bằng Negative Binomial (statsmodels) hoặc LightGBM Poisson.

Bước thử nghiệm (`CLVModel`): OLS hồi quy thẳng lên lợi nhuận gộp, fit trên `slog(y) = sign(y)·log(1+|y|)`, đổi ngược về tiền bằng smearing của Duan (dạng tổng quát:
trung bình của nghịch đảo phép biến đổi trên phần dư train, vì `slog` không phải log thuần nên không có hệ số nhân).
Cấu hình chốt ở notebooks/06_clv.ipynb, lưu tại reports/clv_final.json.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import CATEGORICAL, FEATURES

REPORTS = Path(__file__).resolve().parents[1] / "reports"
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
# Biến đếm / tiền / ngày lệch phải mạnh → lấy slog trước khi đưa vào OLS
SKEWED = ["recency_days", "frequency", "monetary", "tenure_days", "avg_order_value", "mean_gap_days", "std_gap_days",
          "recency_ratio", "orders_3m", "orders_6m", "orders_12m", "active_years", "items_per_order", "n_categories",
          "avg_installments"]
# Biến NaN có nghĩa (chưa có chu kỳ, chưa đánh giá, chưa nhận hàng) → điền 0 + cờ báo thiếu; quy tắc cố định, không học từ dữ liệu
MISSING_FLAGS = {"no_gap": "mean_gap_days", "no_gap_std": "std_gap_days", "no_review": "avg_rating", "no_delivery": "avg_delivery_days"}
SMEAR_N = 2000  # số phần dư train giữ lại để smearing


def slog(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def inv_slog(z):
    z = np.asarray(z, float)
    return np.sign(z) * np.expm1(np.abs(z))


def smear(z_hat, resid) -> np.ndarray:
    """Duan: E[y|x] ≈ mean_j inv_slog(ẑ + e_j)."""
    z_hat = np.asarray(z_hat, float)
    return np.concatenate([inv_slog(c[:, None] + resid[None, :]).mean(1) for c in np.array_split(z_hat, max(1, len(z_hat) // 5000))])


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Ma trận biến cho OLS: slog cho biến lệch, cờ thiếu, dummy (bỏ mức đầu) cho biến phân loại."""
    X = df[NUMERIC].astype(float).copy()
    for flag, col in MISSING_FLAGS.items():
        X[flag] = X[col].isna().astype(float)
    X[SKEWED] = slog(X[SKEWED])
    X = X.fillna(0.0)
    dummies = pd.get_dummies(df[CATEGORICAL], drop_first=True, dtype=float)
    return pd.concat([X, dummies], axis=1)


class CLVModel:
    """OLS trên slog(lợi nhuận gộp); `predict(df)` trả về tiền (smearing) × `scale`."""

    def __init__(self, model, cols: list[str], resid: np.ndarray, scale: float = 1.0):
        self.model, self.cols, self.resid, self.scale = model, cols, resid, scale

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.scale * smear(self.predict_slog(df), self.resid)

    def predict_slog(self, df: pd.DataFrame) -> np.ndarray:
        p = self.model.params
        return design(df)[self.cols].values @ p[self.cols].values + p["const"]


def fit_ols(train: pd.DataFrame, cols: list[str]) -> CLVModel:
    """`train` chỉ gồm khách có quay lại (churn = 0)."""
    res = sm.OLS(slog(train.future_gross_profit), sm.add_constant(design(train)[cols])).fit(cov_type="HC3")
    resid = np.random.RandomState(0).choice(res.resid.values, SMEAR_N, replace=False)
    return CLVModel(res, cols, resid)


class OrdersCLV:
    """`expected_orders(df)` = E[số đơn 12 tháng tới | quay lại]; `predict(df)` = CLV_giữ_được theo tiền.
    `scale` = hệ số hiệu chỉnh mức tổng, fit trên validation (mặt bằng lợi nhuận mỗi đơn đổi theo năm)."""

    def __init__(self, kind: str, model, cols: list[str], value: dict, scale: float = 1.0):
        self.kind, self.model, self.cols, self.value, self.scale = kind, model, cols, value, scale

    def expected_orders(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "nb":
            return 1 + np.asarray(self.model.predict(sm.add_constant(design(df)[self.cols], has_constant="add")))
        return 1 + self.model.predict(df[self.cols])

    def order_value(self, df: pd.DataFrame) -> np.ndarray:
        return df.main_category.astype(str).map(self.value).fillna(self.value["_all"]).values

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.scale * self.expected_orders(df) * self.order_value(df)


def order_value_by_category(train: pd.DataFrame) -> dict:
    """Lợi nhuận giá gốc TB mỗi đơn tương lai = Σ lợi nhuận / Σ số đơn, theo danh mục chủ đạo; `_all` cho danh mục lạ."""
    g = train.groupby(train.main_category.astype(str))
    v = (g.future_profit_full_price.sum() / g.future_orders.sum()).to_dict()
    return {**v, "_all": train.future_profit_full_price.sum() / train.future_orders.sum()}


def fit_nb(train: pd.DataFrame, cols: list[str]) -> OrdersCLV:
    """`train` chỉ gồm khách quay lại. Mục tiêu = số đơn thêm (future_orders − 1), sai số chuẩn robust HC0."""
    res = sm.NegativeBinomial(train.future_orders.values - 1, sm.add_constant(design(train)[cols])).fit(
        disp=0, maxiter=300, cov_type="HC0")
    return OrdersCLV("nb", res, cols, order_value_by_category(train))


def fit_lgbm_poisson(train: pd.DataFrame, cols: list[str], params: dict) -> OrdersCLV:
    m = lgb.LGBMRegressor(objective="poisson", **params, random_state=0, verbose=-1, n_jobs=-1)
    return OrdersCLV("lgbm_poisson", m.fit(train[cols], train.future_orders - 1), cols, order_value_by_category(train))


def fit_final(allowed: set[str] | None = None) -> OrdersCLV:
    """`allowed` khác None: bản web (Task 14), chỉ giữ feature trong tập này (và dummy main_category_*), fit lại `scale` trên validation."""
    from churn_data import load_churn_sets
    cfg = json.loads((REPORTS / "clv_final.json").read_text(encoding="utf8"))
    train, va = load_churn_sets()
    train = train[train.churn == 0]
    cols = cfg["features"] if allowed is None else [c for c in cfg["features"] if c in allowed or c.startswith("main_category_")]
    m = fit_nb(train, cols) if cfg["model"] == "nb" else fit_lgbm_poisson(train, cols, cfg["params"])
    if allowed is None:
        m.scale = cfg["scale"]
    else:
        va = va[va.churn == 0]
        m.scale = float(va.future_profit_full_price.sum() / m.predict(va).sum())
    return m
