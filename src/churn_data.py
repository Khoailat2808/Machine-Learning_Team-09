"""Nạp snapshot churn đã cache (train gộp, validation) với category đồng bộ. Dùng cho notebook 03 trở đi."""
import json
from pathlib import Path

import pandas as pd

from config import TRAIN_CUTOFFS, VALID_CUTOFF
from features import CATEGORICAL

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
FINAL_FEATURES = json.loads((Path(__file__).resolve().parents[1] / "reports" / "churn_features.json")
                            .read_text(encoding="utf8"))["final_features"]


def load_churn_sets():
    """Trả về (train_gộp, validation). Không đọc snapshot test."""
    tr = [pd.read_parquet(INTERIM / f"snap_{c:%Y-%m}.parquet") for c in TRAIN_CUTOFFS]
    va = pd.read_parquet(INTERIM / f"snap_{VALID_CUTOFF:%Y-%m}.parquet")
    for c in CATEGORICAL:
        cats = sorted(set().union(*[set(d[c].dropna().astype(str)) for d in [*tr, va]]))
        for d in [*tr, va]:
            d[c] = pd.Categorical(d[c].astype(str).where(d[c].notna()), categories=cats)
    return pd.concat(tr, ignore_index=True), va
