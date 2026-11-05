"""Đọc CSV DATATHON một lần, lưu parquet vào data/interim/ để notebook sau đọc nhanh.

Chạy: .venv/Scripts/python src/to_parquet.py  (từ gốc repo)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "datathon"
INTERIM = ROOT / "data" / "interim"
SKIP = {"sample_submission"}  # file nộp bài của cuộc thi, không phải dữ liệu


def load_tables(path: Path = INTERIM) -> dict[str, pd.DataFrame]:
    """Đọc lại toàn bộ parquet thành dict {tên bảng: DataFrame}."""
    return {p.stem: pd.read_parquet(p) for p in sorted(path.glob("*.parquet"))}


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    for csv in sorted(RAW.glob("*.csv")):
        if csv.stem in SKIP:
            continue
        df = pd.read_csv(csv, low_memory=False)  # promo_id_2 gần như toàn rỗng, đọc theo khối sẽ đoán sai kiểu
        for col in df.columns:
            if col.lower() == "date" or col.endswith("_date"):
                df[col] = pd.to_datetime(df[col])
        df.to_parquet(INTERIM / f"{csv.stem}.parquet", index=False)
        print(f"{csv.stem:12s} {len(df):>9,} rows  {df.shape[1]:>2} cols")


if __name__ == "__main__":
    main()
