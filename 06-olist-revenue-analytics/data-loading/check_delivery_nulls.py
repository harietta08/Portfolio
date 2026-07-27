"""
Follow-up check: do null delivery dates correspond to non-'delivered' order statuses?
Run from project root with venv activated:
    python data-loading/check_delivery_nulls.py
"""
import pandas as pd
import os

RAW_DIR = os.path.join("data-loading", "raw")
orders = pd.read_csv(os.path.join(RAW_DIR, "olist_orders_dataset.csv"))

print("Order status breakdown (all orders):")
print(orders["order_status"].value_counts())
print()

print("=" * 80)
print("Order status breakdown WHERE order_delivered_customer_date IS NULL:")
null_delivered = orders[orders["order_delivered_customer_date"].isnull()]
print(null_delivered["order_status"].value_counts())
print(f"\nTotal rows with null delivered date: {len(null_delivered)}")
print()

print("=" * 80)
print("Order status breakdown WHERE order_approved_at IS NULL:")
null_approved = orders[orders["order_approved_at"].isnull()]
print(null_approved["order_status"].value_counts())
print(f"\nTotal rows with null approved_at: {len(null_approved)}")
print()

print("=" * 80)
print("Sanity check: any status='delivered' orders with a NULL delivered_customer_date?")
weird = orders[(orders["order_status"] == "delivered") & (orders["order_delivered_customer_date"].isnull())]
print(f"Count: {len(weird)}")
if len(weird) > 0:
    print(weird[["order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date"]].head(10))
