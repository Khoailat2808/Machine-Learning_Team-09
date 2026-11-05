"""Định nghĩa churn và lịch mốc cắt, chốt ở notebooks/02_eda.ipynb (Task 2)."""
import pandas as pd

W_MONTHS = 12  # cửa sổ nhãn: churn = không có đơn hợp lệ trong [t, t + W)
L_YEARS = 3    # tập khách xét: có ít nhất 1 đơn hợp lệ trong [t − L, t)

# Cửa sổ nhãn của train phải kết thúc trước mốc validation, của validation trước mốc test.
TRAIN_CUTOFFS = [pd.Timestamp(d) for d in ("2019-01-01", "2019-07-01", "2020-01-01")]
VALID_CUTOFF = pd.Timestamp("2021-01-01")
TEST_CUTOFF = pd.Timestamp("2022-01-01")  # chỉ chạy một lần ở cuối
DATA_END = pd.Timestamp("2022-12-31")
