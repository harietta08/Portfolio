# =============================================================================
# 02_silver_cleaning.py
# Layer:   Silver — Cleaned, deduplicated, enriched
# Purpose: Read Bronze Delta tables, clean nulls, deduplicate, flatten nested
#          structs, sessionize events, and MERGE into Silver Delta tables.
#          MERGE is the core pattern here — handles late-arriving data and
#          re-runs without creating duplicates (idempotent).
# Run on:  Databricks Community Edition cluster (DBR 13.x+)
# =============================================================================

import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, trim, lower, upper, when, coalesce,
    to_timestamp, to_date, explode, from_json,
    current_timestamp, datediff, row_number,
    sum as spark_sum, count as spark_count,
    min as spark_min, max as spark_max,
    window, lag, unix_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, ArrayType, TimestampType
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BRONZE_DB  = "ecommerce_bronze"
SILVER_DB  = "ecommerce_silver"
SILVER_PATH = "/FileStore/ecommerce/silver"

# -----------------------------------------------------------------------------
# SPARK SESSION
# -----------------------------------------------------------------------------
try:
    spark  # noqa — already exists in Databricks
    print("Running in Databricks — using existing SparkSession")
except NameError:
    spark = SparkSession.builder \
        .appName("ecommerce_silver_cleaning") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DB}")
print(f"Database ready: {SILVER_DB}")


# -----------------------------------------------------------------------------
# HELPER: MERGE into Silver Delta table
# This is the most important function in the entire pipeline.
#
# Why MERGE instead of overwrite?
# - Idempotent: running twice on the same data produces identical results
# - Handles late-arriving data: a record that arrives a day late still gets
#   matched to its existing Silver row and updated, not duplicated
# - Preserves history: Delta Lake logs every MERGE operation
#
# Pattern: WHEN MATCHED AND source is newer → UPDATE
#          WHEN NOT MATCHED → INSERT
# -----------------------------------------------------------------------------
def merge_to_silver(source_df, table_name: str, merge_key: str,
                    update_condition: str = None):
    full_table = f"{SILVER_DB}.{table_name}"
    full_path  = f"{SILVER_PATH}/{table_name}"

    if not DeltaTable.isDeltaTable(spark, full_path):
        # First run — create the table
        print(f"  Creating new Silver table: {full_table}")
        source_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("path", full_path) \
            .saveAsTable(full_table)
    else:
        # Subsequent runs — MERGE
        print(f"  MERGEing into existing Silver table: {full_table}")
        delta_table = DeltaTable.forPath(spark, full_path)

        merge_condition = f"target.{merge_key} = source.{merge_key}"

        (
            delta_table.alias("target")
            .merge(source_df.alias("source"), merge_condition)
            .whenMatchedUpdateAll()   # update all columns if record already exists
            .whenNotMatchedInsertAll() # insert if new record
            .execute()
        )

    count = spark.table(full_table).count()
    print(f"  {full_table} — {count} rows after MERGE")
    return count


# -----------------------------------------------------------------------------
# SILVER: Products
# Cleaning steps:
#   1. Cast rating_rate and rating_count to proper numeric types
#   2. Trim whitespace from title and category
#   3. Standardize category to lowercase
#   4. Drop records with null id or price <= 0
#   5. Add silver audit columns
# -----------------------------------------------------------------------------
def clean_products():
    print("\n[1/3] Cleaning products -> Silver...")

    df = spark.table(f"{BRONZE_DB}.raw_products")

    # Deduplicate — keep latest ingestion per product id
    window_spec = Window.partitionBy("id").orderBy(col("_ingestion_timestamp").desc())
    df = df.withColumn("_row_num", row_number().over(window_spec)) \
           .filter(col("_row_num") == 1) \
           .drop("_row_num")

    df_clean = (
        df
        # Cast types
        .withColumn("id",           col("id").cast(IntegerType()))
        .withColumn("price",        col("price").cast(DoubleType()))
        .withColumn("rating_rate",  col("rating_rate").cast(DoubleType()))
        .withColumn("rating_count", col("rating_count").cast(IntegerType()))

        # Clean strings
        .withColumn("title",    trim(col("title")))
        .withColumn("category", lower(trim(col("category"))))

        # Drop invalid records
        .filter(col("id").isNotNull())
        .filter(col("price") > 0)
        .filter(col("title").isNotNull())

        # Silver audit columns
        .withColumn("_silver_timestamp", current_timestamp())
        .withColumn("_silver_version",   lit("1.0"))

        # Drop Bronze-only columns not needed in Silver
        .drop("_source_endpoint")
    )

    print(f"  Bronze rows: {df.count()} | Silver rows after cleaning: {df_clean.count()}")
    merge_to_silver(df_clean, "products", "id")
    return df_clean


# -----------------------------------------------------------------------------
# SILVER: Users
# Cleaning steps:
#   1. Trim and standardize name fields
#   2. Lowercase email
#   3. Drop password column — never store passwords in analytics layer
#   4. Validate zipcode format
# -----------------------------------------------------------------------------
def clean_users():
    print("\n[2/3] Cleaning users -> Silver...")

    df = spark.table(f"{BRONZE_DB}.raw_users")

    # Deduplicate
    window_spec = Window.partitionBy("id").orderBy(col("_ingestion_timestamp").desc())
    df = df.withColumn("_row_num", row_number().over(window_spec)) \
           .filter(col("_row_num") == 1) \
           .drop("_row_num")

    df_clean = (
        df
        .withColumn("id", col("id").cast(IntegerType()))

        # Clean name fields
        .withColumn("name_firstname", trim(col("name_firstname")))
        .withColumn("name_lastname",  trim(col("name_lastname")))
        .withColumn("full_name",
                    trim(col("name_firstname") + lit(" ") + col("name_lastname")))

        # Standardize email
        .withColumn("email", lower(trim(col("email"))))

        # Clean address
        .withColumn("address_city",    trim(col("address_city")))
        .withColumn("address_zipcode", trim(col("address_zipcode")))

        # Drop sensitive columns — passwords have no place in analytics
        .drop("password")

        # Drop invalid records
        .filter(col("id").isNotNull())
        .filter(col("email").isNotNull())
        .filter(col("username").isNotNull())

        # Silver audit
        .withColumn("_silver_timestamp", current_timestamp())
        .withColumn("_silver_version",   lit("1.0"))
        .drop("_source_endpoint")
    )

    print(f"  Bronze rows: {df.count()} | Silver rows after cleaning: {df_clean.count()}")
    merge_to_silver(df_clean, "users", "id")
    return df_clean


# -----------------------------------------------------------------------------
# SILVER: Cart Items (exploded from raw_carts)
# This is the most complex transformation:
#   1. Parse products_json string back into array of structs
#   2. Explode array — one row per cart line item
#   3. Join with products to get price at time of purchase
#   4. Calculate line_total = quantity * price
#   5. MERGE on composite key (cart_id, product_id)
# -----------------------------------------------------------------------------
def clean_carts():
    print("\n[3/3] Cleaning carts -> Silver (exploding line items)...")

    df = spark.table(f"{BRONZE_DB}.raw_carts")

    # Schema for the nested products_json array
    products_schema = ArrayType(StructType([
        StructField("productId", IntegerType(), True),
        StructField("quantity",  IntegerType(), True)
    ]))

    # Parse the JSON string back to array, then explode to one row per item
    df_exploded = (
        df
        .withColumn("products_array", from_json(col("products_json"), products_schema))
        .withColumn("line_item", explode(col("products_array")))
        .select(
            col("id").cast(IntegerType()).alias("cart_id"),
            col("userId").cast(IntegerType()).alias("user_id"),
            to_timestamp(col("date")).alias("order_timestamp"),
            to_date(col("date")).alias("order_date"),
            col("line_item.productId").cast(IntegerType()).alias("product_id"),
            col("line_item.quantity").cast(IntegerType()).alias("quantity"),
            col("_ingestion_timestamp"),
            col("_ingestion_date")
        )
    )

    # Join with Silver products to get price
    df_products = spark.table(f"{SILVER_DB}.products") \
                       .select("id", "price", "category") \
                       .withColumnRenamed("id", "prod_id")

    df_enriched = (
        df_exploded
        .join(df_products, df_exploded.product_id == df_products.prod_id, "left")
        .drop("prod_id")
        .withColumn("line_total",
                    when(col("price").isNotNull(), col("quantity") * col("price"))
                    .otherwise(lit(None).cast(DoubleType())))
    )

    # Deduplicate on composite key
    window_spec = Window.partitionBy("cart_id", "product_id") \
                        .orderBy(col("_ingestion_timestamp").desc())
    df_clean = df_enriched \
        .withColumn("_row_num", row_number().over(window_spec)) \
        .filter(col("_row_num") == 1) \
        .drop("_row_num") \
        .filter(col("cart_id").isNotNull()) \
        .filter(col("quantity") > 0) \
        .withColumn("_silver_timestamp", current_timestamp()) \
        .withColumn("_silver_version",   lit("1.0")) \
        .drop("_source_endpoint")

    # Composite merge key
    print(f"  Exploded line items: {df_clean.count()}")

    full_table = f"{SILVER_DB}.cart_items"
    full_path  = f"{SILVER_PATH}/cart_items"

    if not DeltaTable.isDeltaTable(spark, full_path):
        df_clean.write.format("delta").mode("overwrite") \
            .option("path", full_path).saveAsTable(full_table)
    else:
        delta_table = DeltaTable.forPath(spark, full_path)
        (
            delta_table.alias("target")
            .merge(
                df_clean.alias("source"),
                "target.cart_id = source.cart_id AND target.product_id = source.product_id"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    count = spark.table(full_table).count()
    print(f"  {full_table} — {count} rows after MERGE")
    return df_clean


# -----------------------------------------------------------------------------
# SILVER: Order summaries (cart-level aggregation for Gold prep)
# -----------------------------------------------------------------------------
def build_order_summaries():
    print("\n[+] Building order summaries...")

    df_items = spark.table(f"{SILVER_DB}.cart_items")

    df_orders = (
        df_items.groupBy("cart_id", "user_id", "order_date", "order_timestamp")
        .agg(
            spark_count("product_id").alias("item_count"),
            spark_sum("quantity").alias("total_quantity"),
            spark_sum("line_total").alias("order_value")
        )
        .withColumn("_silver_timestamp", current_timestamp())
        .withColumn("_silver_version",   lit("1.0"))
    )

    full_table = f"{SILVER_DB}.orders"
    full_path  = f"{SILVER_PATH}/orders"

    if not DeltaTable.isDeltaTable(spark, full_path):
        df_orders.write.format("delta").mode("overwrite") \
            .option("path", full_path).saveAsTable(full_table)
    else:
        delta_table = DeltaTable.forPath(spark, full_path)
        (
            delta_table.alias("target")
            .merge(df_orders.alias("source"),
                   "target.cart_id = source.cart_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    count = spark.table(full_table).count()
    print(f"  {full_table} — {count} rows")
    return df_orders


# -----------------------------------------------------------------------------
# TIME TRAVEL DEMONSTRATION
# Run this after executing the script a second time to see two versions.
# Paste into a new Databricks cell to demonstrate Delta time travel.
# -----------------------------------------------------------------------------
TIME_TRAVEL_DEMO = """
-- Delta Lake Time Travel Examples
-- Run these in a Databricks SQL cell after running this script at least once

-- See all versions of the products table
DESCRIBE HISTORY ecommerce_silver.products;

-- Query the first version (before any updates)
SELECT * FROM ecommerce_silver.products VERSION AS OF 0 LIMIT 5;

-- Query as of a specific timestamp
-- SELECT * FROM ecommerce_silver.products
-- TIMESTAMP AS OF '2024-01-01 00:00:00' LIMIT 5;

-- Compare record counts across versions
SELECT 'version_0' as version, COUNT(*) as cnt
FROM ecommerce_silver.products VERSION AS OF 0
UNION ALL
SELECT 'current' as version, COUNT(*) as cnt
FROM ecommerce_silver.products;
"""


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"Silver Cleaning Start: {datetime.now().isoformat()}")
    print("=" * 60)

    df_products = clean_products()
    df_users    = clean_users()
    df_items    = clean_carts()
    df_orders   = build_order_summaries()

    print("\n" + "=" * 60)
    print("Silver Cleaning Complete — Verification")
    print("=" * 60)

    for table in ["products", "users", "cart_items", "orders"]:
        full = f"{SILVER_DB}.{table}"
        cnt  = spark.table(full).count()
        print(f"  {full:35s} {cnt:>6} rows")

    print("\nSample — silver.cart_items (enriched with price):")
    spark.table(f"{SILVER_DB}.cart_items") \
         .select("cart_id", "user_id", "product_id",
                 "quantity", "price", "line_total", "order_date") \
         .show(5)

    print("\nSample — silver.orders:")
    spark.table(f"{SILVER_DB}.orders").show(5)

    print("\nTime travel demo SQL saved to TIME_TRAVEL_DEMO variable.")
    print("Paste into a Databricks SQL cell to demonstrate Delta time travel.")
    print(f"\nSilver Cleaning End: {datetime.now().isoformat()}")
    print("Next step: Run 03_gold_aggregation.py")


if __name__ == "__main__":
    main()
