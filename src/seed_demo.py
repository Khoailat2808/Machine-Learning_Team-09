"""Seed dữ liệu demo DATATHON vào MongoDB của Web (Task 14). Chỉ thêm; mọi bản ghi có `demoTag: 'datathon'`.

Chạy từ gốc repo (cần MONGODB_URI, DB mặc định lấy từ URI):
    .venv/Scripts/python src/seed_demo.py seed [--customers 2000]
    .venv/Scripts/python src/seed_demo.py purge      # xóa đúng các bản ghi có demoTag, không đụng dữ liệu khác
Cần vai trò CUSTOMER đã có trong `roles` (Web seed sẵn). _id sinh tất định từ id gốc nên seed hai lần sẽ báo trùng, không nhân đôi.
"""
import argparse
import hashlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from bson import ObjectId

TAG = "datathon"
COLLECTIONS = ["users", "categories", "products", "product_variants", "orders"]
# trạng thái dataset -> trạng thái Web (web_adapter đảo lại; chỉ cần giữ đúng ranh giới đơn hợp lệ / không hợp lệ)
STATUS = {"created": "PENDING_CONFIRMATION", "paid": "CONFIRMED", "shipped": "SHIPPING", "delivered": "DELIVERED",
          "returned": "RETURNING", "cancelled": "CANCELLED"}
GENDER = {"Male": "MALE", "Female": "FEMALE"}


def oid(kind: str, key) -> ObjectId:
    return ObjectId(hashlib.md5(f"{TAG}:{kind}:{key}".encode()).digest()[:12])


def _dt(ts) -> datetime:
    return pd.Timestamp(ts).to_pydatetime()


def sample_customers(tables: dict, n: int, since: str = "2020-01-01", seed: int = 0) -> np.ndarray:
    """n khách ngẫu nhiên (cố định theo seed) có đơn hợp lệ từ `since`: đủ gần để vào cửa sổ L năm khi chấm điểm."""
    o = tables["orders"]
    ids = np.sort(o[(o.order_date >= since) & ~o.order_status.isin(["cancelled", "created"])].customer_id.unique())
    return np.random.RandomState(seed).choice(ids, min(n, len(ids)), replace=False)


def build_docs(tables: dict, customer_ids, role_id: ObjectId) -> dict[str, list[dict]]:
    """Dữ liệu thô (schema DATATHON) -> document theo schema Mongoose của Web. Không đụng DB."""
    o = tables["orders"][tables["orders"].customer_id.isin(customer_ids)]
    it = tables["order_items"][tables["order_items"].order_id.isin(o.order_id)]
    pr = tables["products"][tables["products"].product_id.isin(it.product_id)]
    cu = tables["customers"].set_index("customer_id").loc[list(customer_ids)]
    first = o.groupby("customer_id").order_date.min()
    tag = {"demoTag": TAG}

    users = [{"_id": oid("user", cid), "roleId": role_id, "email": f"demo-{cid}@datathon.local", "authSource": "EMAIL",
              "profile": {"fullName": f"Khách demo {cid}", "gender": GENDER.get(r.gender, "OTHER"), "dateOfBirth": None, "birthDateEditCount": 0},
              "avatar": "", "termsAccepted": True, "status": "ACTIVE",
              "security": {"failedLoginCount": 0, "lockCount": 0, "lockedUntil": None, "mustChangePassword": False},
              "lockInfo": {"reason": None, "lockedBy": None, "lockedAt": None}, "anonymizedAt": None,
              "createdAt": _dt(first[cid]), "updatedAt": _dt(first[cid]), "demoRef": int(cid), **tag}
             for cid, r in cu.iterrows()]
    cats = [{"_id": oid("category", c), "parentId": None, "name": c, "slug": f"datathon-{c.lower()}", "type": "PRODUCT_CATEGORY",
             "description": "", "image": "", "status": "HIDDEN", "typicalCategoryIds": [], **tag} for c in sorted(pr.category.unique())]
    # Mỗi sản phẩm DATATHON = 1 product + 1 variant; ẩn khỏi cửa hàng (HIDDEN / DISCONTINUED)
    products = [{"_id": oid("product", p.product_id), "name": p.product_name, "slug": f"datathon-p{p.product_id}", "productCode": f"DT{p.product_id}",
                 "tags": [], "categoryIds": [oid("category", p.category)], "manualBadges": {"isNew": None, "isBestSeller": None},
                 "isAutoBestSeller": False, "warrantyMonths": 0, "status": "HIDDEN", "media": [], "parts": [], **tag} for p in pr.itertuples()]
    variants = [{"_id": oid("variant", p.product_id), "productId": oid("product", p.product_id), "sku": f"DT-{p.product_id}", "packages": [],
                 "partDimensions": [], "originalPrice": float(p.price), "salePrice": float(p.price), "stockQuantity": 0, "lowStockThreshold": 0,
                 "installationService": "NONE", "status": "DISCONTINUED", **tag} for p in pr.itertuples()]

    lines = {k: g for k, g in it.groupby("order_id")}
    orders = []
    for r in o.itertuples():
        g = lines.get(r.order_id)
        items = [{"variantId": oid("variant", l.product_id), "snapshot": {"name": f"DT-{l.product_id}", "sku": f"DT-{l.product_id}"},
                  "quantity": int(l.quantity), "unitPrice": float(l.unit_price),
                  "promotionId": oid("promo", l.promo_id) if isinstance(l.promo_id, str) else None,
                  "promotionDiscount": float(l.discount_amount), "voucherDiscountShare": 0.0,
                  "finalAmount": float(l.quantity * l.unit_price - l.discount_amount), "warrantyMonths": 0}
                 for l in (g.itertuples() if g is not None else [])]
        sub = sum(i["quantity"] * i["unitPrice"] for i in items)
        disc = sum(i["promotionDiscount"] for i in items)
        status = STATUS[r.order_status]
        orders.append({"_id": oid("order", r.order_id), "orderCode": f"DT{r.order_id}", "userId": oid("user", r.customer_id),
                       "idempotencyKey": f"datathon-{r.order_id}",
                       "shippingAddress": {"recipientName": "Demo", "recipientPhone": "0000000000", "addressLine": "Demo", "ward": "Demo", "province": "Demo"},
                       "subtotal": sub, "promotionDiscount": disc, "voucherDiscount": 0.0, "shippingFee": 0.0, "installationFee": 0.0,
                       "totalAmount": sub - disc, "status": status, "items": items,
                       "statusHistory": [{"changedBy": None, "toStatus": status, "changedAt": _dt(r.order_date)}],
                       "payment": {"method": "COD" if r.payment_method == "cod" else "BANK_TRANSFER", "collectionType": "FULL",
                                   "status": "UNPAID" if status in ("CANCELLED", "PENDING_CONFIRMATION") else "PAID"},
                       "source": "CART", "invoiceInfo": {"type": "NONE", "sendEmail": False},
                       "createdAt": _dt(r.order_date), "updatedAt": _dt(r.order_date), **tag})
    return {"users": users, "categories": cats, "products": products, "product_variants": variants, "orders": orders}


def seed(db, tables: dict, n_customers: int = 2000, seed_: int = 0) -> dict[str, int]:
    role = db.roles.find_one({"name": "CUSTOMER"})
    if role is None:
        raise SystemExit("Chưa có vai trò CUSTOMER trong `roles`: chạy seed của Web trước.")
    if db.users.count_documents({"demoTag": TAG}):
        raise SystemExit("Đã có dữ liệu demo: chạy `purge` trước nếu muốn seed lại.")
    docs = build_docs(tables, sample_customers(tables, n_customers, seed=seed_), role["_id"])
    for name in COLLECTIONS:  # theo thứ tự phụ thuộc; ordered=False để một dòng lỗi không chặn cả lô
        if docs[name]:
            db[name].insert_many(docs[name], ordered=False)
    return {k: len(v) for k, v in docs.items()}


def purge(db) -> dict[str, int]:
    return {name: db[name].delete_many({"demoTag": TAG}).deleted_count for name in COLLECTIONS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["seed", "purge"])
    ap.add_argument("--customers", type=int, default=2000)
    a = ap.parse_args()
    from pymongo import MongoClient
    uri = os.environ.get("MONGODB_URI") or sys.exit("Thiếu biến môi trường MONGODB_URI")
    database = MongoClient(uri).get_default_database()
    if a.cmd == "seed":
        from to_parquet import load_tables
        print(seed(database, load_tables(), a.customers))
    else:
        print(purge(database))
