# =============================================================================
# 03_gold_aggregation.py
# Layer:   Gold — Customer-level KPIs, ready for BigQuery and dbt
# Purpose: Aggregate Silver tables into business-ready metrics.
#          One row per customer. Optimized for query performance.
#          This is what dbt reads from BigQuery after export.
# Run on:  Databricks Community Edition cluster (DBR 13.x+)
# =============================================================================

from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, coalesce, current_timestamp,
    datediff, to_date, countDistinct,
    sum as spark_sum, count as spark_count,
    min as spark_min, max as spark_max,
    avg as spark_avg, round as spark_round,
    first, last, row_number
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SILVER_DB   = "ecommerce_silver"
GOLD_DB     = "ecommerce_gold"
GOLD_PATH   = "/FileStore/ecommerce/gold"
AS_OF_DATE  = date.today().isoformat()

# -----------------------------------------------------------------------------
# SPARK SESSION
# -----------------------------------------------------------------------------
try:
    spark  # noqa
    print("Running in Databricks — using existing SparkSession")
except NameError:
    spark = SparkSession.builder \
        .appName("ecommerce_gold_aggregation") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_DB}")
print(f"Database ready: {GOLD_DB}")


# -----------------------------------------------------------------------------
# GOLD: Customer KPIs
# One row per customer. The core table that dbt mart models read from.
#
# Metrics:
#   - total_orders:         how many orders placed
#   - total_spend:          lifetime revenue from this customer
#   - avg_order_value:      total_spend / total_orders
#   - total_items:          total products purchased
#   - unique_products:      distinct product SKUs purchased
#   - first_order_date:     acquisition date (for cohort analysis)
#   - last_order_date:      recency (for churn scoring)
#   - days_since_last_order: recency in days (as of today)
#   - customer_lifespan_days: days between first and last order
#   - order_frequency:      orders per month (for LTV projection)
#   - ltv_simple:           total_spend (historical LTV)
#   - ltv_projected_12m:    projected 12-month LTV based on frequency
#   - customer_segment:     High/Mid/Low value based on spend percentile
# -----------------------------------------------------------------------------
def build_customer_kpis():
    print("\n[1/2] Building customer KPIs...")

    df_orders  = spark.table(f"{SILVER_DB}.orders")
    df_items   = spark.table(f"{SILVER_DB}.cart_items")
    df_users   = spark.table(f"{SILVER_DB}.users")

    # Base customer metrics from orders
    df_base = (
        df_orders.groupBy("user_id")
        .agg(
            spark_count("cart_id").alias("total_orders"),
            spark_sum("order_value").alias("total_spend"),
            spark_avg("order_value").alias("avg_order_value"),
            spark_sum("total_quantity").alias("total_items"),
            spark_min("order_date").alias("first_order_date"),
            spark_max("order_date").alias("last_order_date")
        )
    )

    # Unique products purchased
    df_unique_products = (
        df_items.groupBy("user_id")
        .agg(countDistinct("product_id").alias("unique_products"))
    )

    # Favourite category (most purchased by line_total)
    window_cat = Window.partitionBy("user_id").orderBy(col("cat_spend").desc())
    df_fav_category = (
        df_items.groupBy("user_id", "category")
        .agg(spark_sum("line_total").alias("cat_spend"))
        .withColumn("rank", row_number().over(window_cat))
        .filter(col("rank") == 1)
        .select("user_id", col("category").alias("favourite_category"))
    )

    # Join all metrics together
    df_customer = (
        df_base
        .join(df_unique_products, "user_id", "left")
        .join(df_fav_category,    "user_id", "left")
        .join(df_users.select("id", "username", "email",
                              "address_city", "name_firstname", "name_lastname"),
              df_base.user_id == df_users.id, "left")
        .drop(df_users.id)
    )

    # Derived metrics
    df_customer = (
        df_customer
        .withColumn("days_since_last_order",
                    datediff(lit(AS_OF_DATE).cast("date"), col("last_order_date")))
        .withColumn("customer_lifespan_days",
                    datediff(col("last_order_date"), col("first_order_date")))
        # Orders per month — avoid division by zero for single-order customers
        .withColumn("order_frequency_monthly",
                    when(col("customer_lifespan_days") > 30,
                         spark_round(col("total_orders") /
                                     (col("customer_lifespan_days") / 30.0), 2))
                    .otherwise(lit(1.0).cast(DoubleType())))
        # Simple LTV = total spend to date
        .withColumn("ltv_simple",
                    spark_round(col("total_spend"), 2))
        # Projected 12-month LTV = avg_order_value * projected_orders_next_12m
        .withColumn("ltv_projected_12m",
                    spark_round(col("avg_order_value") *
                                col("order_frequency_monthly") * 12, 2))
        # Round monetary columns
        .withColumn("total_spend",      spark_round(col("total_spend"), 2))
        .withColumn("avg_order_value",  spark_round(col("avg_order_value"), 2))
    )

    # Customer segment based on total_spend percentile
    # Using simple thresholds for CE — in production use approxQuantile
    df_customer = (
        df_customer
        .withColumn("customer_segment",
                    when(col("total_spend") >= 500, lit("high_value"))
                    .when(col("total_spend") >= 200, lit("mid_value"))
                    .otherwise(lit("low_value")))
        # Churn risk — no order in last 60 days
        .withColumn("is_churned",
                    when(col("days_since_last_order") > 60, lit(True))
                    .otherwise(lit(False)))
        .withColumn("_gold_timestamp", current_timestamp())
        .withColumn("_gold_version",   lit("1.0"))
        .withColumn("_as_of_date",     lit(AS_OF_DATE))
    )

    # Write Gold — full refresh (idempotent: same input = same output)
    full_table = f"{GOLD_DB}.customer_kpis"
    full_path  = f"{GOLD_PATH}/customer_kpis"

    df_customer.write \
        .format("delta") \
        .mode("overwrite") \
        .option("path", full_path) \
        .option("overwriteSchema", "true") \
        .saveAsTable(full_table)

    count = spark.table(full_table).count()
    print(f"  {full_table} — {count} customer rows")
    return df_customer


# -----------------------------------------------------------------------------
# GOLD: Product performance metrics
# Aggregated product-level stats for the dashboard category tab
# -----------------------------------------------------------------------------
def build_product_metrics():
    print("\n[2/2] Building product metrics...")

    df_items    = spark.table(f"{SILVER_DB}.cart_items")
    df_products = spark.table(f"{SILVER_DB}.products")

    df_product_metrics = (
        df_items.groupBy("product_id")
        .agg(
            spark_count("cart_id").alias("times_ordered"),
            spark_sum("quantity").alias("total_units_sold"),
            spark_sum("line_total").alias("total_revenue"),
            countDistinct("user_id").alias("unique_buyers"),
            spark_avg("quantity").alias("avg_quantity_per_order")
        )
        .join(df_products.select("id", "title", "category", "price", "rating_rate"),
              df_items.product_id == df_products.id, "left")
        .drop(df_products.id)
        .withColumn("total_revenue",   spark_round(col("total_revenue"), 2))
        .withColumn("avg_quantity_per_order", spark_round(col("avg_quantity_per_order"), 2))
        .withColumn("revenue_per_buyer",
                    spark_round(col("total_revenue") / col("unique_buyers"), 2))
        .withColumn("_gold_timestamp", current_timestamp())
        .withColumn("_gold_version",   lit("1.0"))
        .withColumn("_as_of_date",     lit(AS_OF_DATE))
    )

    full_table = f"{GOLD_DB}.product_metrics"
    full_path  = f"{GOLD_PATH}/product_metrics"

    df_product_metrics.write \
        .format("delta") \
        .mode("overwrite") \
        .option("path", full_path) \
        .option("overwriteSchema", "true") \
        .saveAsTable(full_table)

    count = spark.table(full_table).count()
    print(f"  {full_table} — {count} product rows")
    return df_product_metrics


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"Gold Aggregation Start: {datetime.now().isoformat()}")
    print(f"As-of date: {AS_OF_DATE}")
    print("=" * 60)

    df_customers = build_customer_kpis()
    df_products  = build_product_metrics()

    print("\n" + "=" * 60)
    print("Gold Aggregation Complete — Verification")
    print("=" * 60)

    for table in ["customer_kpis", "product_metrics"]:
        full = f"{GOLD_DB}.{table}"
        cnt  = spark.table(full).count()
        print(f"  {full:35s} {cnt:>6} rows")

    print("\nGold customer KPIs sample:")
    spark.table(f"{GOLD_DB}.customer_kpis") \
         .select("user_id", "username", "total_orders", "total_spend",
                 "avg_order_value", "ltv_projected_12m",
                 "customer_segment", "is_churned", "days_since_last_order") \
         .show(10, truncate=False)

    print("\nGold product metrics sample:")
    spark.table(f"{GOLD_DB}.product_metrics") \
         .select("product_id", "title", "category",
                 "times_ordered", "total_revenue", "unique_buyers") \
         .orderBy(col("total_revenue").desc()) \
         .show(10, truncate=60)

    print(f"\nGold Aggregation End: {datetime.now().isoformat()}")
    print("Next step: Run export/gold_to_bigquery.py")


if __name__ == "__main__":
    main()
