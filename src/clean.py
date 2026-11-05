"""Quy tắc làm sạch DATATHON, rút ra từ notebooks/01_data_audit.ipynb.

Không xóa dòng nào: chỉ thêm cột đã chuẩn hóa và bỏ cột không đáng tin,
để các feature như tỷ lệ hủy vẫn tính được từ đơn bị loại.
"""
import pandas as pd

INVALID_STATUSES = {"cancelled", "created"}  # đơn chưa thành giao dịch


def clean_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    t = dict(tables)

    # R1. Đơn hợp lệ = khác cancelled/created; returned vẫn là một lần mua.
    orders = t["orders"].copy()
    orders["is_valid"] = ~orders["order_status"].isin(INVALID_STATUSES)
    t["orders"] = orders

    # R2. Doanh thu và giá vốn cấp dòng. Giữ dòng trùng (order, product):
    # payment_value bằng đúng tổng các dòng kể cả dòng trùng.
    items = t["order_items"].merge(t["products"][["product_id", "cogs"]], on="product_id", how="left")
    items["line_revenue"] = items["quantity"] * items["unit_price"] - items["discount_amount"]
    items["line_cogs"] = items["quantity"] * items["cogs"]

    # R3. Tiền hoàn không vượt doanh thu thực của dòng hàng đó (11% bản ghi vượt).
    line_rev = items.groupby(["order_id", "product_id"], as_index=False)["line_revenue"].sum()
    returns = t["returns"].merge(line_rev, on=["order_id", "product_id"], how="left")
    returns["refund_capped"] = returns["refund_amount"].clip(upper=returns["line_revenue"])
    t["returns"] = returns.drop(columns="line_revenue")

    # R11. Lợi nhuận gộp dòng = doanh thu − tiền hoàn − giá vốn của phần KHÔNG bị trả
    # (giả định hàng trả về bán lại được). Với 16 cặp dòng trùng, gán trả hàng cho dòng đầu.
    ret = returns.groupby(["order_id", "product_id"], as_index=False).agg(
        return_qty=("return_quantity", "sum"), refund=("refund_capped", "sum"))
    first = items.groupby(["order_id", "product_id"]).cumcount() == 0
    items = items.merge(ret, on=["order_id", "product_id"], how="left")
    items[["return_qty", "refund"]] = items[["return_qty", "refund"]].fillna(0)
    items.loc[~first.values, ["return_qty", "refund"]] = 0  # merge trái giữ nguyên thứ tự dòng
    kept = (items["quantity"] - items["return_qty"]).clip(lower=0)
    items["gross_profit"] = items["line_revenue"] - items["refund"] - kept * items["line_cogs"] / items["quantity"]
    t["order_items"] = items.drop(columns="cogs")

    # R4. signup_date không tương quan với đơn đầu tiên (r = 0,002) -> bỏ;
    # thâm niên tính từ đơn hợp lệ đầu tiên.
    t["customers"] = t["customers"].drop(columns="signup_date")
    return t
