# ML: churn, CLV (DATATHON 2026)

Website tích hợp mô hình (trang quản trị Marketing): [WEBADVANCE](https://github.com/xinmotlansay/WEBADVANCE).

## Dữ liệu

- Nguồn: DATATHON 2026 (e-commerce thời trang Việt Nam, 2012–2022). Link tải: **TODO: dán link trang cuộc thi**.
- Giải nén các file CSV vào `data/raw/datathon/` (`customers.csv`, `orders.csv`, ...).
- `data/` không commit (đã có trong `.gitignore`).

## Cài đặt (Windows, Python 3.11)

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/to_parquet.py   # CSV -> data/interim/*.parquet (chạy một lần)
.venv/Scripts/python -m ipykernel install --user --name ml-venv --display-name "Python 3.11 (ml/.venv)"
```

Notebook dùng kernel `ml-venv` (venv ở trên), không dùng kernel `python3` mặc định.

Trong notebook đọc dữ liệu bằng:

```python
import sys; sys.path.insert(0, "../src")
from to_parquet import load_tables
from clean import clean_tables
tables = clean_tables(load_tables())   # {"orders": DataFrame, ...}, đã áp quy tắc làm sạch
```

## Cấu trúc

| Thư mục | Nội dung |
| :--- | :--- |
| `data/raw/datathon/` | CSV gốc |
| `data/interim/` | parquet đã ép kiểu ngày |
| `notebooks/` | notebook trình bày, chạy theo số thứ tự |
| `src/` | code dùng lại (`to_parquet.py`, `clean.py`, `features.py`, `predict.py`) |
| `models/` | artifact `.joblib`, `metadata.json` |

## Thứ tự chạy notebook

1. `01_data_audit.ipynb`: kiểm tra toàn vẹn dữ liệu
2. `02_eda.ipynb`: EDA, chốt định nghĩa churn
3. `03_churn_features.ipynb`: lọc biến churn
4. `04_churn_model.ipynb`: so sánh model churn
   - `04b_churn_tuning_eval.ipynb`: tinh chỉnh, hiệu chỉnh, đánh giá test
5. `05_churn_explain.ipynb`: SHAP, phân tích lỗi
6. `06_clv.ipynb`: hồi quy CLV hai phần
7. `07_business.ipynb`: mô phỏng lợi nhuận, ma trận phân khúc

Test chống rò rỉ: `.venv/Scripts/python -m pytest src`.

## Tích hợp Web (Giai đoạn B)

Cần `MONGODB_URI` (DB lấy từ URI). Chạy từ gốc repo, sau `python src/predict.py build web`:

```bash
.venv/Scripts/python src/seed_demo.py seed --customers 2000   # thêm dữ liệu demo (demoTag='datathon'); purge để xóa
.venv/Scripts/python src/score_web.py                         # chấm điểm, upsert customer_scores (--cutoff YYYY-MM-DD)
.venv/Scripts/python src/export_demo.py ../WEBADVANCE/server/seed/data/ml-demo  # ghi {users,orders,customer_scores}.json vào seed của repo Web (bỏ tham số: ghi ra export/)
cd src && ../.venv/Scripts/python -m uvicorn api:app --host 127.0.0.1 --port 8000   # service: GET /health, POST /score (409 nếu đang chạy)
```

`POST /score` cần header `X-ML-Token` trùng biến `ML_API_TOKEN` của service: sai hoặc thiếu token trả 401; service chưa đặt `ML_API_TOKEN` trả 503 (fail closed). Server Web đặt cùng giá trị ở `ML_API_TOKEN` trong `.env` và tự gửi header. `/health` không cần token.

### Hợp đồng ML ↔ Web (`schemaVersion` 1; bản gốc ở repo Web: `WEBADVANCE/tasks/marketing-admin-plan.md`)

Collection `customer_scores`, mỗi khách một document, khóa `customerId` (ObjectId = `_id` của `users`):
`pChurn` (0–1), `clv`, `segment` (`RETAIN` | `NURTURE` | `NO_SPEND` | `LET_BE`), `topReasons` (≤ 3 phần tử `{feature, direction: UP|DOWN}`; UP = tăng rủi ro),
`net`, `sStar` (null nếu khách không đáng voucher: vô cùng), `voucherCost`, `isControl` (hash `customerId`, ~10% mỗi phân khúc, giữ nguyên giữa các lần chấm),
`scoredAt`, `cutoff`, `modelVersion`, `schemaVersion`.
API: `GET /health` → `{ok, modelVersion}`; `POST /score` → `{scored, scoredAt, modelVersion}` (đồng bộ).
