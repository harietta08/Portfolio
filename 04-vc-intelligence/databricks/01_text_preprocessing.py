# ── databricks/01_text_preprocessing.py ───────────────────────────────────────
# Runs in Databricks Community Edition
# Purpose: Clean and normalize startup description text using Spark
# Why Spark: Parallelizes text cleaning across thousands of records
# In interviews: "Databricks owns training. FastAPI owns serving. Intentionally separated."

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, length, udf
)
from pyspark.sql.types import StringType
import mlflow
import re

# ── Start MLflow run ──────────────────────────────────────────────────────────
mlflow.set_experiment("/vc-intelligence/text-preprocessing")

spark = SparkSession.builder.appName("vc-text-preprocessing").getOrCreate()

# ── Load raw data from GCS ────────────────────────────────────────────────────
# In CE: upload CSV to Databricks FileStore and read from there
# df = spark.read.csv("gs://vc-intelligence-data/raw/startups.csv", header=True)
df = spark.read.csv("/FileStore/tables/startups_sample.csv", header=True, inferSchema=True)

print(f"Records loaded: {df.count()}")
df.printSchema()

# ── Text cleaning UDF ─────────────────────────────────────────────────────────
def clean_text(text):
    """
    Normalize startup description text.
    Decisions:
    - Lowercase: reduces vocab size for TF-IDF
    - Remove special chars: reduces noise
    - Collapse whitespace: clean token boundaries
    - Keep numbers: funding amounts are signal
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\.\,]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

clean_udf = udf(clean_text, StringType())

# ── Apply cleaning ────────────────────────────────────────────────────────────
df_clean = df.withColumn("description_clean", clean_udf(col("description")))

# Remove records with empty descriptions after cleaning
df_clean = df_clean.filter(length(col("description_clean")) > 10)

print(f"Records after cleaning: {df_clean.count()}")

# ── Log stats to MLflow ───────────────────────────────────────────────────────
with mlflow.start_run(run_name="text_preprocessing"):
    mlflow.log_param("cleaning_steps", "lowercase,remove_special_chars,collapse_whitespace")
    mlflow.log_param("min_description_length", 10)
    mlflow.log_metric("records_input", df.count())
    mlflow.log_metric("records_output", df_clean.count())
    mlflow.log_metric("records_dropped", df.count() - df_clean.count())

# ── Save processed data ───────────────────────────────────────────────────────
df_clean.write.mode("overwrite").parquet("/FileStore/tables/startups_clean")
print("Saved to /FileStore/tables/startups_clean")

df_clean.select("name", "description", "description_clean", "sector", "stage").show(5, truncate=60)
