"""
Foreign key integrity check across Olist tables.
Run from project root with venv activated:
    python data-loading/check_foreign_keys.py
"""
import pandas as pd
import os

RAW_DIR = os.path.join("data-loading", "raw")

def load(name):
    return pd.read_csv(os.path.join(RAW_DIR, name))

orders = load("olist_orders_dataset.csv")
order_items = load("olist_order_items_dataset.csv")
order_payments = load("olist_order_payments_dataset.csv")
order_reviews = load("olist_order_reviews_dataset.csv")
customers = load("olist_customers_dataset.csv")
products = load("olist_products_dataset.csv")
sellers = load("olist_sellers_dataset.csv")
category_translation = load("product_category_name_translation.csv")

def check_fk(child_df, child_col, parent_df, parent_col, child_name, parent_name):
    child_ids = set(child_df[child_col].dropna().unique())
    parent_ids = set(parent_df[parent_col].dropna().unique())
    missing = child_ids - parent_ids
    print(f"{child_name}.{child_col} -> {parent_name}.{parent_col}")
    print(f"  Unique {child_col} values in {child_name}: {len(child_ids)}")
    print(f"  Missing from {parent_name}: {len(missing)}")
    if missing:
        print(f"  Example missing IDs: {list(missing)[:5]}")
    print()

print("=" * 80)
print("FOREIGN KEY INTEGRITY CHECKS")
print("=" * 80)

check_fk(orders, "customer_id", customers, "customer_id", "orders", "customers")
check_fk(order_items, "order_id", orders, "order_id", "order_items", "orders")
check_fk(order_items, "product_id", products, "product_id", "order_items", "products")
check_fk(order_items, "seller_id", sellers, "seller_id", "order_items", "sellers")
check_fk(order_payments, "order_id", orders, "order_id", "order_payments", "orders")
check_fk(order_reviews, "order_id", orders, "order_id", "order_reviews", "orders")
check_fk(products, "product_category_name", category_translation, "product_category_name",
         "products", "category_translation")

print("=" * 80)
print("REVERSE CHECK: orders with NO matching order_items (orphaned orders)")
orders_with_items = set(order_items["order_id"].unique())
all_orders = set(orders["order_id"].unique())
orphaned = all_orders - orders_with_items
print(f"  Orders with zero order_items rows: {len(orphaned)}")
if orphaned:
    sample_orphans = orders[orders["order_id"].isin(list(orphaned)[:5])]
    print(f"  Example order_status for orphaned orders:")
    print(orders[orders["order_id"].isin(orphaned)]["order_status"].value_counts())
print()

print("=" * 80)
print("DUPLICATE order_id CHECK (should be unique per order in 'orders' table)")
dup_orders = orders["order_id"].duplicated().sum()
print(f"  Duplicate order_id rows in 'orders' table: {dup_orders}")

print()
print("Done.")
