"""Model churn cuối (cấu hình đóng băng ở reports/churn_final.json) và `top_reasons` cho web.

Fit lại từ train + hiệu chỉnh sigmoid trên validation, tất định nên cho đúng kết quả của notebook 04b.
Task 11 sẽ lưu thành joblib; đến lúc đó hàm này chỉ còn dùng khi huấn luyện lại.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn_data import FINAL_FEATURES, load_churn_sets

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# feature -> (nhãn hiển thị, định dạng giá trị)
LABELS = {
    "frequency": ("Số đơn đã mua", "{:.0f}"),
    "recency_days": ("Số ngày kể từ lần mua cuối", "{:.0f}"),
    "monetary": ("Tổng chi tiêu", "{:,.0f}"),
    "mean_gap_days": ("Chu kỳ mua trung bình (ngày)", "{:.0f}"),
    "std_gap_days": ("Độ lệch chu kỳ mua (ngày)", "{:.0f}"),
    "review_rate": ("Tỷ lệ đơn có đánh giá", "{:.0%}"),
    "cancel_rate": ("Tỷ lệ hủy đơn", "{:.0%}"),
    "avg_installments": ("Số kỳ trả góp trung bình", "{:.1f}"),
    "n_categories": ("Số danh mục đã mua", "{:.0f}"),
    "main_source": ("Kênh đặt đơn chủ đạo", "{}"),
    "main_category": ("Danh mục chủ đạo", "{}"),
    "region": ("Vùng miền", "{}"),
}

# Vẫn nằm trong model nhưng không hiển thị làm lý do: tác động trái trực giác (từng hủy đơn → churn ít hơn)
# và nghi rò rỉ nhẹ vì `order_status` là trạng thái cuối (Task 5, 7).
HIDDEN_REASONS = {"cancel_rate"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ChurnModel:
    def __init__(self, booster: lgb.LGBMClassifier, calibrator: LogisticRegression, features: list[str]):
        self.model, self.cal, self.features = booster, calibrator, features

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Xác suất churn đã hiệu chỉnh."""
        return self.cal.predict_proba(_logit(self.predict_raw(X))[:, None])[:, 1]

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Đóng góp từng biến vào log-odds thô (tổng + cột `bias` = log-odds thô). Dương = tăng rủi ro churn."""
        c = self.model.booster_.predict(X[self.features], pred_contrib=True)
        return pd.DataFrame(c, index=X.index, columns=[*self.features, "bias"])

    def top_reasons(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Với mỗi khách (dòng của X): k biến tác động mạnh nhất, kèm giá trị và hướng tác động."""
        contrib = self.contributions(X)[[f for f in self.features if f not in HIDDEN_REASONS]]
        out = []
        for i in range(len(X)):
            row = contrib.iloc[i]
            reasons = []
            for f in row.abs().sort_values(ascending=False).index[:k]:
                label, fmt = LABELS[f]
                v = X[f].iloc[i]
                reasons.append({"feature": f, "label": label,
                                "value": "không có" if pd.isna(v) else fmt.format(v),
                                "effect": "tăng rủi ro" if row[f] > 0 else "giảm rủi ro",
                                "contribution": float(row[f])})
            out.append(reasons)
        return out


def fit_final(features: list[str] | None = None) -> ChurnModel:
    """`features` khác None: bản rút gọn cho web (Task 14), cùng tham số đóng băng."""
    cfg = json.loads((REPORTS / "churn_final.json").read_text(encoding="utf8"))
    train, va = load_churn_sets()
    F = features or FINAL_FEATURES
    m = lgb.LGBMClassifier(**cfg["params"], subsample_freq=1, random_state=0, verbose=-1, n_jobs=-1,
                           class_weight=cfg["class_weight"]).fit(train[F], train.churn)
    cal = LogisticRegression(C=1e6).fit(_logit(m.predict_proba(va[F])[:, 1])[:, None], va.churn)
    return ChurnModel(m, cal, F)
