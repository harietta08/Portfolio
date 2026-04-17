# =============================================================================
# 01_bronze_ingestion.py
# Layer:   Bronze — Raw ingestion into Delta Lake
# Purpose: Land raw API data as-is into Delta tables. No transformations.
#          Append-only. Full history preserved. Auto schema detection.
# Run on:  Databricks Community Edition cluster (DBR 13.x+)
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import requests
import json
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, current_timestamp, to_date, input_file_name
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType
)
from delta.tables import DeltaTable

# -----------------------------------------------------------------------------
# CONFIG
# Note: In Databricks CE, set these as notebook widgets or cluster env vars.
# In production, use Databricks Secrets: dbutils.secrets.get("scope", "key")
# -----------------------------------------------------------------------------
BRONZE_DB        = "ecommerce_bronze"
BRONZE_PATH      = "/FileStore/ecommerce/bronze"   # DBFS path in CE
API_BASE_URL     = "https://fakestoreapi.com"
INGESTION_DATE   = date.today().isoformat()         # partition key: YYYY-MM-DD

# -----------------------------------------------------------------------------
# SPARK SESSION
# In Databricks, SparkSession is already available as `spark`.
# This block makes the script runnable locally for testing too.
# -----------------------------------------------------------------------------
try:
    spark  # noqa: F821 — already exists in Databricks
    print("Running in Databricks — using existing SparkSession")
except NameError:
    spark = SparkSession.builder \
        .appName("ecommerce_bronze_ingestion") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    print("Running locally — created new SparkSession")

spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_DB}")
print(f"Database ready: {BRONZE_DB}")


# -----------------------------------------------------------------------------
# HELPER: Fetch from API
# -----------------------------------------------------------------------------
def fetch_api(endpoint: str) -> list:
    """
    Fetch a Fake Store API endpoint and return raw JSON as a list of dicts.
    Raises on non-200 status so the pipeline fails loudly instead of
    silently landing corrupt data.
    """
    url = f"{API_BASE_URL}/{endpoint}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Fetched /{endpoint} — {len(data)} records")
    return data


# -----------------------------------------------------------------------------
# HELPER: Add Bronze metadata columns
# -----------------------------------------------------------------------------
def add_bronze_metadata(df, source_endpoint: str):
    """
    Every Bronze record gets three audit columns:
    - _ingestion_timestamp: when this record landed in the lake
    - _source_endpoint:     which API endpoint it came from
    - _ingestion_date:      partition column for date-based pruning
    These columns are never modified in Silver or Gold — they are the
    immutable audit trail of when data entered the system.
    """
    return (
        df
        .withColumn("_ingestion_timestamp", current_timestamp())
        .withColumn("_source_endpoint",     lit(source_endpoint))
        .withColumn("_ingestion_date",      lit(INGESTION_DATE))
    )


# -----------------------------------------------------------------------------
# HELPER: Write to Delta Bronze (append-only)
# -----------------------------------------------------------------------------
def write_bronze(df, table_name: str, partition_col: str = "_ingestion_date"):
    """
    Append raw records to a Delta table partitioned by ingestion date.

    Why append-only?
    Bronze is the immutable raw layer. We never update or delete here.
    If the source sends bad data, we can always time-travel back to see
    exactly what arrived and when. Silver handles deduplication via MERGE.
    """
    full_table = f"{BRONZE_DB}.{table_name}"
    full_path  = f"{BRONZE_PATH}/{table_name}"

    (
        df.write
        .format("delta")
        .mode("append")
        .partitionBy(partition_col)
        .option("path", full_path)
        .option("mergeSchema", "true")   # schema evolution: new API columns land safely
        .saveAsTable(full_table)
    )

    count = spark.table(full_table).count()
    print(f"  Wrote {df.count()} rows -> {full_table} (total rows: {count})")


# -----------------------------------------------------------------------------
# INGEST: Products
# Schema: id, title, price, description, category, image, rating{rate,count}
# Note:   rating is a nested struct — we stringify it at Bronze and explode
#         it properly in Silver. Bronze never transforms nested structures.
# -----------------------------------------------------------------------------
def ingest_products():
    print("\n[1/3] Ingesting products...")
    raw = fetch_api("products")

    # Flatten rating struct to strings at Bronze layer
    # Silver will parse these into proper numeric columns
    for record in raw:
        if isinstance(record.get("rating"), dict):
            record["rating_rate"]  = str(record["rating"].get("rate", ""))
            record["rating_count"] = str(record["rating"].get("count", ""))
            del record["rating"]

    df = spark.createDataFrame(raw)
    df = df.withColumn("id",    col("id").cast(IntegerType())) \
           .withColumn("price", col("price").cast(DoubleType()))
    df = add_bronze_metadata(df, "products")
    write_bronze(df, "raw_products")
    return df


# -----------------------------------------------------------------------------
# INGEST: Users
# Schema: id, email, username, password, name{}, address{}, phone
# Note:   name and address are nested structs. Same pattern — stringify at
#         Bronze, parse in Silver.
# Privacy note: In production, PII (email, name, address) would be masked
#               or tokenized before landing in Bronze. Noted for portfolio.
# -----------------------------------------------------------------------------
def ingest_users():
    print("\n[2/3] Ingesting users...")
    raw = fetch_api("users")

    for record in raw:
        # Flatten nested name struct
        if isinstance(record.get("name"), dict):
            record["name_firstname"] = record["name"].get("firstname", "")
            record["name_lastname"]  = record["name"].get("lastname", "")
            del record["name"]

        # Flatten nested address struct
        if isinstance(record.get("address"), dict):
            addr = record["address"]
            record["address_city"]    = addr.get("city", "")
            record["address_street"]  = addr.get("street", "")
            record["address_number"]  = str(addr.get("number", ""))
            record["address_zipcode"] = addr.get("zipcode", "")
            # Drop geolocation — not needed for analytics
            del record["address"]

    df = spark.createDataFrame(raw)
    df = df.withColumn("id", col("id").cast(IntegerType()))
    df = add_bronze_metadata(df, "users")
    write_bronze(df, "raw_users")
    return df


# -----------------------------------------------------------------------------
# INGEST: Carts (Orders)
# Schema: id, userId, date, products[{productId, quantity}]
# Note:   products is a nested array of structs. At Bronze we store the raw
#         JSON string. Silver uses explode() to create one row per line item.
# -----------------------------------------------------------------------------
def ingest_carts():
    print("\n[3/3] Ingesting carts...")
    raw = fetch_api("carts")

    for record in raw:
        # Serialize nested products array to JSON string at Bronze
        # Silver will explode this into individual cart line items
        if isinstance(record.get("products"), list):
            record["products_json"] = json.dumps(record["products"])
            del record["products"]

    df = spark.createDataFrame(raw)
    df = df.withColumn("id",     col("id").cast(IntegerType())) \
           .withColumn("userId", col("userId").cast(IntegerType()))
    df = add_bronze_metadata(df, "carts")
    write_bronze(df, "raw_carts")
    return df


# -----------------------------------------------------------------------------
# MAIN — Run all ingestion jobs
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"Bronze Ingestion Start: {datetime.now().isoformat()}")
    print(f"Ingestion date:         {INGESTION_DATE}")
    print(f"Target database:        {BRONZE_DB}")
    print("=" * 60)

    df_products = ingest_products()
    df_users    = ingest_users()
    df_carts    = ingest_carts()

    # ------------------------------------------------------------------
    # VERIFICATION: Show table counts and sample records
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Bronze Ingestion Complete — Verification")
    print("=" * 60)

    for table in ["raw_products", "raw_users", "raw_carts"]:
        full_table = f"{BRONZE_DB}.{table}"
        count      = spark.table(full_table).count()
        print(f"  {full_table:35s} {count:>6} rows")

    # Show one sample record from each table
    print("\nSample — raw_products:")
    spark.table(f"{BRONZE_DB}.raw_products").limit(2).show(truncate=60)

    print("Sample — raw_users:")
    spark.table(f"{BRONZE_DB}.raw_users") \
         .select("id", "username", "email", "_ingestion_date") \
         .limit(2).show(truncate=40)

    print("Sample — raw_carts:")
    spark.table(f"{BRONZE_DB}.raw_carts").limit(2).show(truncate=60)

    # ------------------------------------------------------------------
    # DELTA LAKE HISTORY — demonstrates time travel capability
    # After running this script twice, you can query:
    #   SELECT * FROM ecommerce_bronze.raw_products VERSION AS OF 0
    # to see the first ingestion. This is documented in the README.
    # ------------------------------------------------------------------
    print("\nDelta history — raw_products:")
    spark.sql(f"DESCRIBE HISTORY {BRONZE_DB}.raw_products") \
         .select("version", "timestamp", "operation") \
         .show(5, truncate=False)

    print(f"\nBronze Ingestion End: {datetime.now().isoformat()}")
    print("Next step: Run 02_silver_cleaning.py")


if __name__ == "__main__":
    main()
