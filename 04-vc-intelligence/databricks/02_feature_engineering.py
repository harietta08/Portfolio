# ── databricks/02_feature_engineering.py ──────────────────────────────────────
# Runs in Databricks Community Edition
# Purpose: Build TF-IDF features and sentence embeddings, log to MLflow
# Why both: TF-IDF = interpretable (SHAP), embeddings = semantic (ChromaDB search)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import mlflow
import numpy as np
import pandas as pd
import pickle

mlflow.set_experiment("/vc-intelligence/feature-engineering")

spark = SparkSession.builder.appName("vc-feature-engineering").getOrCreate()

# ── Load cleaned data ─────────────────────────────────────────────────────────
df_spark = spark.read.parquet("/FileStore/tables/startups_clean")
df = df_spark.toPandas()
print(f"Records: {len(df)}")

# ── TF-IDF features ───────────────────────────────────────────────────────────
# max_features=5000: covers domain vocab without noise
# ngram_range=(1,2): captures "machine learning", "climate tech" as single features
# Why TF-IDF: interpretable via SHAP — can show which words drive sector prediction
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(df["description_clean"])
print(f"TF-IDF matrix: {X_tfidf.shape}")

# ── Sentence embeddings ───────────────────────────────────────────────────────
# all-MiniLM-L6-v2: fast, 384-dim, strong semantic similarity performance
# Why embeddings: captures "carbon capture" ~ "climate tech" relationships
# TF-IDF misses — used for ChromaDB semantic search
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["description_clean"].tolist(), show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# ── Log to MLflow ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="feature_engineering"):
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("tfidf_ngram_range", "(1,2)")
    mlflow.log_param("tfidf_min_df", 2)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("embedding_dim", embeddings.shape[1])
    mlflow.log_metric("vocab_size", len(tfidf.vocabulary_))
    mlflow.log_metric("num_records", len(df))

    # Save vectorizer as artifact
    with open("/tmp/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    mlflow.log_artifact("/tmp/tfidf_vectorizer.pkl")

    # Save embeddings as artifact
    np.save("/tmp/embeddings.npy", embeddings)
    mlflow.log_artifact("/tmp/embeddings.npy")

print("Features logged to MLflow")

# ── Save locally for next step ────────────────────────────────────────────────
df["embedding"] = embeddings.tolist()
df.to_parquet("/FileStore/tables/startups_features.parquet", index=False)
print("Saved: /FileStore/tables/startups_features.parquet")
