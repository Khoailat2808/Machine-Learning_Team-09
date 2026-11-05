"""Tạo snapshot cấp khách tại một mốc cắt: feature (chỉ từ dữ liệu trước mốc) + nhãn.

Nguyên tắc chống rò rỉ: mọi sự kiện có ngày riêng (trả hàng, giao hàng, đánh giá) chỉ được
dùng khi ngày đó < mốc cắt. Không dùng `order_items.refund` / `gross_profit` cho feature vì
chúng gộp cả tiền hoàn phát sinh sau mốc. Kiểm chứng bằng src/test_features.py.

Giới hạn đã biết: `order_status` là trạng thái cuối cùng, không có lịch sử. Một đơn đặt sát mốc
rồi bị hủy sau mốc vẫn bị coi là đã hủy. Ảnh hưởng nhỏ (hủy thường xảy ra ngay), ghi vào báo cáo.
"""
import numpy as np
import pandas as pd

from config import L_YEARS, W_MONTHS

LATE_DELIVERY_DAYS = 8  # đặt → nhận > 8 ngày (phân phối 2–10 ngày, ~11% đơn trên mức này)

# Từ điển feature: tên -> (nhóm, công thức, ý nghĩa). Là nguồn duy nhất cho phụ lục báo cáo.
FEATURES = {
    # RFM
    "recency_days": ("RFM", "mốc cắt − ngày đơn hợp lệ cuối", "Bao lâu rồi khách chưa mua"),
    "frequency": ("RFM", "số đơn hợp lệ trước mốc", "Tổng số lần mua"),
    "monetary": ("RFM", "Σ doanh thu dòng − Σ tiền hoàn (hoàn trước mốc)", "Tổng tiền khách đã chi thực"),
    "tenure_days": ("RFM", "mốc cắt − ngày đơn hợp lệ đầu", "Thâm niên (thay cho signup_date bị nhiễu)"),
    "avg_order_value": ("RFM", "Σ doanh thu dòng / frequency", "Giá trị đơn trung bình"),
    # Nhịp mua
    "mean_gap_days": ("Nhịp mua", "trung bình khoảng cách giữa hai đơn liên tiếp (NaN nếu 1 đơn)", "Chu kỳ mua quen thuộc"),
    "std_gap_days": ("Nhịp mua", "độ lệch chuẩn khoảng cách (NaN nếu < 3 đơn)", "Độ đều đặn khi mua"),
    "recency_ratio": ("Nhịp mua", "recency_days / mean_gap_days", "Khách đang 'trễ hẹn' bao nhiêu chu kỳ"),
    "orders_3m": ("Nhịp mua", "số đơn trong 3 tháng trước mốc", "Hoạt động rất gần đây"),
    "orders_6m": ("Nhịp mua", "số đơn trong 6 tháng trước mốc", "Hoạt động gần đây"),
    "orders_12m": ("Nhịp mua", "số đơn trong 12 tháng trước mốc", "Hoạt động năm vừa qua"),
    "spend_trend": ("Nhịp mua", "log(1 + chi 6 tháng gần nhất) − log(1 + chi 6 tháng trước đó)", "Chi tiêu đang tăng (> 0) hay giảm (< 0)"),
    "active_years": ("Nhịp mua", "số năm dương lịch có ít nhất 1 đơn", "Độ bền của quan hệ"),
    # Hành vi
    "promo_share": ("Hành vi", "tỷ lệ đơn có ít nhất 1 dòng khuyến mãi", "Độ ưa khuyến mãi"),
    "avg_discount_rate": ("Hành vi", "Σ tiền giảm / Σ (số lượng × đơn giá)", "Mức giảm trung bình được hưởng"),
    "return_rate": ("Hành vi", "Σ số lượng trả (trả trước mốc) / Σ số lượng mua", "Tỷ lệ đổi trả"),
    "avg_rating": ("Hành vi", "điểm đánh giá trung bình (đánh giá trước mốc; NaN nếu chưa đánh giá)", "Mức hài lòng"),
    "low_rating_share": ("Hành vi", "tỷ lệ đánh giá ≤ 2 sao", "Trải nghiệm tệ"),
    "review_rate": ("Hành vi", "số đánh giá / frequency", "Mức độ tương tác"),
    "avg_delivery_days": ("Hành vi", "trung bình (ngày nhận − ngày đặt), đơn đã nhận trước mốc", "Tốc độ giao hàng khách trải qua"),
    "late_delivery_share": ("Hành vi", f"tỷ lệ đơn giao > {LATE_DELIVERY_DAYS} ngày", "Trải nghiệm giao trễ"),
    "cancel_rate": ("Hành vi", "số đơn cancelled / tổng số đơn đã đặt (mọi trạng thái)", "Thói quen hủy đơn"),
    "cod_share": ("Hành vi", "tỷ lệ đơn hợp lệ trả bằng COD", "Mức tin tưởng thanh toán trước"),
    "installment_share": ("Hành vi", "tỷ lệ đơn hợp lệ trả góp (> 1 kỳ)", "Dùng trả góp"),
    "avg_installments": ("Hành vi", "số kỳ trả góp trung bình", "Mức phụ thuộc trả góp"),
    "mobile_share": ("Hành vi", "tỷ lệ đơn đặt trên mobile", "Thiết bị"),
    "desktop_share": ("Hành vi", "tỷ lệ đơn đặt trên desktop (tablet = phần còn lại)", "Thiết bị"),
    "main_source": ("Hành vi", "order_source xuất hiện nhiều nhất", "Kênh đặt đơn chủ đạo"),
    "items_per_order": ("Hành vi", "Σ số lượng / frequency", "Quy mô giỏ hàng"),
    "n_categories": ("Hành vi", "số danh mục khác nhau đã mua", "Độ rộng nhu cầu"),
    "main_category": ("Hành vi", "danh mục có doanh thu lớn nhất", "Nhu cầu chủ đạo"),
    "gross_margin_rate": ("Hành vi", "(Σ doanh thu dòng − Σ giá vốn) / Σ doanh thu dòng, trước hoàn tiền", "Khách mua hàng lãi hay hàng lỗ"),
    # Nhân khẩu học
    "gender": ("Nhân khẩu học", "customers.gender", ""),
    "age_group": ("Nhân khẩu học", "customers.age_group", ""),
    "acquisition_channel": ("Nhân khẩu học", "customers.acquisition_channel", "Kênh khách đến lần đầu"),
    "region": ("Nhân khẩu học", "geography.region theo zip của khách", "Vùng miền"),
}
CATEGORICAL = ["main_source", "main_category", "gender", "age_group", "acquisition_channel", "region"]
# future_orders, future_profit_full_price: cho CLV dạng số đơn × lợi nhuận mỗi đơn (Task 9).
# full_price = lợi nhuận gộp cộng lại tiền giảm giá, vì chi phí khuyến mãi được trừ riêng ở bước mô phỏng (Task 10).
LABELS = ["churn", "future_gross_profit", "future_orders", "future_profit_full_price"]


def _mode(df: pd.DataFrame, col: str, weight: str | None = None) -> pd.Series:
    g = df.groupby(["customer_id", col], observed=True)
    s = g[weight].sum() if weight else g.size()
    # Hòa thì lấy giá trị đứng trước theo thứ tự chữ cái, để kết quả tất định
    s = s.reset_index(name="w").sort_values(["customer_id", "w", col], ascending=[True, False, True])
    return s.drop_duplicates("customer_id").set_index("customer_id")[col]


def build_snapshot(t: dict[str, pd.DataFrame], cutoff, w_months: int = W_MONTHS,
                   l_years: int = L_YEARS) -> pd.DataFrame:
    """Mỗi dòng một khách có đơn hợp lệ trong [cutoff − L năm, cutoff). `t` là output của clean_tables."""
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.DateOffset(months=w_months)
    o = t["orders"]

    placed = o[o.order_date < cutoff]
    v = placed[placed.is_valid]
    last = v.groupby("customer_id").order_date.max()
    ids = last[last >= cutoff - pd.DateOffset(years=l_years)].index
    v = v[v.customer_id.isin(ids)].sort_values(["customer_id", "order_date", "order_id"])
    placed = placed[placed.customer_id.isin(ids)]
    f = pd.DataFrame(index=pd.Index(ids, name="customer_id"))

    # Dòng hàng của đơn hợp lệ trước mốc
    it = t["order_items"].merge(v[["order_id", "customer_id", "order_date"]], on="order_id")
    it = it.merge(t["products"][["product_id", "category"]], on="product_id", how="left")
    it["gross"] = it.quantity * it.unit_price
    per_order = it.groupby("order_id").agg(revenue=("line_revenue", "sum"), promo=("promo_id", lambda s: s.notna().any()))
    v = v.join(per_order, on="order_id")

    # Tiền hoàn và số lượng trả: chỉ các lần trả xảy ra trước mốc
    rt = t["returns"][t["returns"].return_date < cutoff].merge(v[["order_id", "customer_id"]], on="order_id")
    refund = rt.groupby("customer_id").refund_capped.sum()

    by = v.groupby("customer_id")
    f["recency_days"] = (cutoff - by.order_date.max()).dt.days
    f["frequency"] = by.size()
    revenue = by.revenue.sum()
    f["monetary"] = revenue - refund.reindex(f.index, fill_value=0)
    f["tenure_days"] = (cutoff - by.order_date.min()).dt.days
    f["avg_order_value"] = revenue / f.frequency

    gap = by.order_date.diff().dt.days
    f["mean_gap_days"] = gap.groupby(v.customer_id).mean()
    f["std_gap_days"] = gap.groupby(v.customer_id).std()
    f["recency_ratio"] = (f.recency_days / f.mean_gap_days).replace([np.inf, -np.inf], np.nan)
    for m in (3, 6, 12):
        f[f"orders_{m}m"] = v[v.order_date >= cutoff - pd.DateOffset(months=m)].groupby("customer_id").size()
    last6 = v[v.order_date >= cutoff - pd.DateOffset(months=6)].groupby("customer_id").revenue.sum()
    prev6 = v[(v.order_date < cutoff - pd.DateOffset(months=6)) &
              (v.order_date >= cutoff - pd.DateOffset(months=12))].groupby("customer_id").revenue.sum()
    f["spend_trend"] = np.log1p(last6.reindex(f.index, fill_value=0)) - np.log1p(prev6.reindex(f.index, fill_value=0))
    f["active_years"] = v.assign(y=v.order_date.dt.year).groupby("customer_id").y.nunique()

    f["promo_share"] = by.promo.mean()
    ic = it.groupby("customer_id")
    f["avg_discount_rate"] = ic.discount_amount.sum() / ic.gross.sum()
    f["return_rate"] = rt.groupby("customer_id").return_quantity.sum().reindex(f.index, fill_value=0) / ic.quantity.sum()

    rv = t["reviews"][(t["reviews"].review_date < cutoff) & t["reviews"].order_id.isin(v.order_id)]
    rb = rv.groupby("customer_id").rating
    f["avg_rating"] = rb.mean()
    f["low_rating_share"] = rb.apply(lambda r: (r <= 2).mean())
    f["review_rate"] = rv.groupby("customer_id").size().reindex(f.index, fill_value=0) / f.frequency

    sh = t["shipments"][t["shipments"].delivery_date < cutoff].merge(v[["order_id", "customer_id", "order_date"]], on="order_id")
    sh["days"] = (sh.delivery_date - sh.order_date).dt.days
    f["avg_delivery_days"] = sh.groupby("customer_id").days.mean()
    f["late_delivery_share"] = (sh.days > LATE_DELIVERY_DAYS).groupby(sh.customer_id).mean()

    f["cancel_rate"] = (placed.order_status == "cancelled").groupby(placed.customer_id).mean()
    f["cod_share"] = (v.payment_method == "cod").groupby(v.customer_id).mean()
    pay = t["payments"][["order_id", "installments"]].merge(v[["order_id", "customer_id"]], on="order_id")
    f["installment_share"] = (pay.installments > 1).groupby(pay.customer_id).mean()
    f["avg_installments"] = pay.groupby("customer_id").installments.mean()
    f["mobile_share"] = (v.device_type == "mobile").groupby(v.customer_id).mean()
    f["desktop_share"] = (v.device_type == "desktop").groupby(v.customer_id).mean()
    f["main_source"] = _mode(v, "order_source")
    f["items_per_order"] = ic.quantity.sum() / f.frequency
    f["n_categories"] = ic.category.nunique()
    f["main_category"] = _mode(it, "category", weight="line_revenue")
    f["gross_margin_rate"] = (ic.line_revenue.sum() - ic.line_cogs.sum()) / ic.line_revenue.sum()

    cust = t["customers"].merge(t["geography"][["zip", "region"]], on="zip", how="left").set_index("customer_id")
    for col in ("gender", "age_group", "acquisition_channel", "region"):
        f[col] = cust[col]

    count_cols = ["orders_3m", "orders_6m", "orders_12m"]
    f[count_cols] = f[count_cols].fillna(0).astype(int)
    for col in CATEGORICAL:
        f[col] = f[col].astype("category")

    # Nhãn: chỉ dùng dữ liệu trong [cutoff, cutoff + W)
    fut = o[(o.order_date >= cutoff) & (o.order_date < end) & o.is_valid & o.customer_id.isin(ids)]
    fut_it = t["order_items"].merge(fut[["order_id", "customer_id"]], on="order_id").groupby("customer_id")
    f["churn"] = (~f.index.isin(fut.customer_id)).astype(int)
    f["future_gross_profit"] = fut_it.gross_profit.sum().reindex(f.index, fill_value=0.0)
    f["future_orders"] = fut.groupby("customer_id").size().reindex(f.index, fill_value=0).astype(int)
    f["future_profit_full_price"] = f.future_gross_profit + fut_it.discount_amount.sum().reindex(f.index, fill_value=0.0)

    f.insert(0, "cutoff", cutoff)
    return f[["cutoff", *FEATURES, *LABELS]]


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame([(k, *v) for k, v in FEATURES.items()], columns=["feature", "nhóm", "công thức", "ý nghĩa"])
