"""Đọc MongoDB của Web, trả về `tables` đúng cấu trúc `clean.clean_tables` của dữ liệu train (Task 14).

    t = web_adapter.load_tables(db)            # db: pymongo Database
    predict.score(t, cutoff, web=True)         # chỉ dùng model web (feature web có)

Khác dataset: không có giá vốn (dùng tỷ suất trung vị theo danh mục), trả hàng, đánh giá, giao hàng, thiết bị, kênh, vùng, trả góp.
Các bảng đó để rỗng/NaN; model web không dùng feature nào từ chúng.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clean import clean_tables

MODELS = Path(__file__).resolve().parents[1] / "models"
# trạng thái Web -> từ vựng dataset (clean.INVALID_STATUSES = cancelled/created: đơn chưa thành giao dịch)
STATUS = {"CANCELLED": "cancelled", "PENDING_PAYMENT": "created", "PENDING_CONFIRMATION": "created", "DELIVERY_FAILED": "created",
          "RETURNING": "returned", "CONFIRMED": "paid", "HANDED_OVER": "shipped", "SHIPPING": "shipped", "DELIVERED": "delivered"}


def _empty(**cols) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})


def load_tables(db, cogs_ratio: dict | None = None) -> dict[str, pd.DataFrame]:
    if cogs_ratio is None:
        cogs_ratio = json.loads((MODELS / "metadata_web.json").read_text(encoding="utf8"))["cogs_ratio"]
    docs = list(db.orders.find({}, {"userId": 1, "status": 1, "createdAt": 1, "source": 1, "payment.method": 1, "totalAmount": 1, "items": 1}))

    orders, lines = [], []
    for d in docs:
        oid = str(d["_id"])
        orders.append((oid, d["createdAt"], str(d["userId"]), -1, STATUS[d["status"]], "cod" if d["payment"]["method"] == "COD" else "bank_transfer",
                       None, d["source"].lower(), d["totalAmount"]))
        for i in d["items"]:
            promo = i.get("promotionId")
            lines.append((oid, str(i["variantId"]), i["quantity"], i["unitPrice"], i.get("promotionDiscount", 0) + i.get("voucherDiscountShare", 0),
                          str(promo) if promo else None))
    cols = ["order_id", "order_date", "customer_id", "zip", "order_status", "payment_method", "device_type", "order_source", "total"]
    orders = pd.DataFrame(orders, columns=cols)
    order_items = pd.DataFrame(lines, columns=["order_id", "product_id", "quantity", "unit_price", "discount_amount", "promo_id"])
    order_items["promo_id_2"] = None
    order_items["promo_id"] = order_items.promo_id.astype(object)

    # variant -> product -> danh mục đầu tiên; giá vốn = giá bán × tỷ suất trung vị của danh mục
    vids = [i["variantId"] for d in docs for i in d["items"]]
    var = {v["_id"]: v for v in db.product_variants.find({"_id": {"$in": vids}}, {"productId": 1, "salePrice": 1, "sku": 1})}
    prod = {p["_id"]: p for p in db.products.find({"_id": {"$in": [v["productId"] for v in var.values()]}}, {"name": 1, "categoryIds": 1})}
    cat = {c["_id"]: c["name"] for c in db.categories.find({"_id": {"$in": [i for p in prod.values() for i in p.get("categoryIds", [])[:1]]}}, {"name": 1})}
    rows = []
    for vid, v in var.items():
        p = prod.get(v["productId"], {})
        c = cat.get((p.get("categoryIds") or [None])[0], "unknown")
        rows.append((str(vid), p.get("name", v["sku"]), c, None, None, None, v["salePrice"], v["salePrice"] * cogs_ratio.get(c, cogs_ratio["_all"])))
    products = pd.DataFrame(rows, columns=["product_id", "product_name", "category", "segment", "size", "color", "price", "cogs"])

    users = orders.customer_id.drop_duplicates()
    customers = pd.DataFrame({"customer_id": users, "zip": -1, "city": None, "signup_date": pd.NaT, "gender": None, "age_group": None,
                              "acquisition_channel": None})
    t = {
        "orders": orders.drop(columns="total"), "order_items": order_items, "products": products, "customers": customers,
        "payments": pd.DataFrame({"order_id": orders.order_id, "payment_method": orders.payment_method, "payment_value": orders.total,
                                  "installments": 1}),
        "geography": _empty(zip="int64", city="object", region="object", district="object"),
        "returns": _empty(return_id="object", order_id="object", product_id="object", return_date="datetime64[ns]", return_reason="object",
                          return_quantity="int64", refund_amount="float64"),
        "reviews": _empty(review_id="object", order_id="object", product_id="object", customer_id="object", review_date="datetime64[ns]", rating="int64"),
        "shipments": _empty(order_id="object", ship_date="datetime64[ns]", delivery_date="datetime64[ns]", shipping_fee="float64"),
    }
    t["orders"]["order_date"] = pd.to_datetime(t["orders"].order_date)
    return clean_tables(t)
