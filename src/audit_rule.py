"""Số liệu đối chiếu ma trận 2×2 với quy tắc net > 0 cho báo cáo (tasks/voucher-rule-plan.md, Task 2).

Chạy từ gốc repo:  .venv/Scripts/python src/audit_rule.py > reports/audit_rule_output.txt
Ba chính sách: phát cả ô Giữ ngay; net > 0 thuần (mốc so sánh của Mục 4.6); ô Giữ ngay và net > 0 (quy tắc của hệ thống).
"""
import pandas as pd

import predict
from clean import clean_tables
from to_parquet import load_tables

S_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
CUTOFFS = {"Chọn mô hình 01/2021": "2021-01-01", "Kiểm tra 01/2022": "2022-01-01"}


def net_at(df: pd.DataFrame, s: float) -> pd.Series:
    return df.p_churn * s * (df.clv - df.voucher_cost) - (1 - df.p_churn) * df.voucher_cost


def policies(df: pd.DataFrame, s: float) -> dict[str, pd.Series]:
    cell, net = df.segment == "Giữ ngay", net_at(df, s)
    return {"Phát cả ô Giữ ngay": cell, "net > 0 thuần": net > 0, "Ô Giữ ngay và net > 0": cell & (net > 0)}


def report(name: str, df: pd.DataFrame) -> None:
    print(f"\n===== {name}: {len(df):,} khách =====")
    net30 = net_at(df, predict.S_REF)
    print("\n-- Bảng chéo ở s = 30% (số khách) --")
    print(pd.crosstab(df.segment, net30 > 0, margins=True).rename(columns={False: "net <= 0", True: "net > 0"}))
    print("\n-- Tổng net ở s = 30% theo ô (triệu) --")
    print((net30.groupby([df.segment, net30 > 0]).sum() / 1e6).round(3).unstack().rename(columns={False: "net <= 0", True: "net > 0"}))
    print("\n-- Tỷ lệ khách được phát / lợi nhuận kỳ vọng (triệu) --")
    rows = []
    for s in S_GRID:
        net = net_at(df, s)
        rows.append({"s": f"{s:.0%}", **{k: f"{m.mean():.1%} / {net[m].sum() / 1e6:+.2f}" for k, m in policies(df, s).items()}})
    print(pd.DataFrame(rows).to_string(index=False))
    send = df.send_voucher
    print(f"\nsend_voucher (s = 30%): {send.sum():,} khách ({send.mean():.1%}); trong ô Giữ ngay: "
          f"{send.sum() / (df.segment == 'Giữ ngay').sum():.1%} số khách của ô")


if __name__ == "__main__":
    tables = clean_tables(load_tables())
    print(f"p_cut = {predict.load()[2]['p_cut']:.4f}, s_ref = {predict.S_REF}")
    for name, cutoff in CUTOFFS.items():
        report(name, predict.score(tables, cutoff))
