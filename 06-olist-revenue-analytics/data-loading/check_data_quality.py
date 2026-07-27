"""
Quick data quality check for Olist dataset CSVs.
Run this from the project root (06-olist-revenue-analytics) with venv activated:
    python data-loading/check_data_quality.py
"""
import pandas as pd
import os

RAW_DIR = os.path.join("data-loading", "raw")

files = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

for f in files:
    path = os.path.join(RAW_DIR, f)
    print("=" * 80)
    print(f"FILE: {f}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        continue

    print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Duplicate rows (exact): {df.duplicated().sum()}")

    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print("  Columns with nulls:")
        for col, cnt in null_cols.items():
            pct = round(100 * cnt / len(df), 2)
            print(f"    {col}: {cnt} nulls ({pct}%)")
    else:
        print("  No nulls found.")
    print()

print("=" * 80)
print("Done. Review output above for anomalies before proceeding to table creation.")
