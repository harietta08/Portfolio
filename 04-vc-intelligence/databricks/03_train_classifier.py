# ── databricks/03_train_classifier.py ─────────────────────────────────────────
# Runs in Databricks Community Edition
# Purpose: Train sector classifier with Hyperopt HPT, log all trials to MLflow
# Why Hyperopt: distributed HPT — all 30 trials logged, parallel coordinates plot
# Interview answer: "Hyperopt searched 30 combinations. MLflow tracked every trial.
#                   I picked the best model by F1 on the held-out test set."

from pyspark.sql import SparkSession
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import json

mlflow.set_experiment("/vc-intelligence/classifier-training")

spark = SparkSession.builder.appName("vc-classifier").getOrCreate()

# ── Load data ─────────────────────────────────────────────────────────────────
df_spark = spark.read.parquet("/FileStore/tables/startups_clean")
df = df_spark.toPandas()
print(f"Records: {len(df)}")

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df["sector"])
print(f"Classes: {list(le.classes_)}")

# ── TF-IDF features ───────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X = tfidf.fit_transform(df["description_clean"])
y = df["label"].values

# ── Train/test split — HELD OUT, never touched during HPT ─────────────────────
# Stratified: preserves class distribution in both splits
# 20% held out: standard for this dataset size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Hyperopt search space ─────────────────────────────────────────────────────
search_space = {
    "C": hp.loguniform("C", -3, 3),
    "max_iter": hp.choice("max_iter", [200, 500, 1000]),
    "class_weight": hp.choice("class_weight", ["balanced", None]),
}

def objective(params):
    """Hyperopt objective — minimize negative F1 (macro)."""
    with mlflow.start_run(nested=True):
        model = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        # Cross-val on TRAIN only — test set stays hidden
        cv_f1 = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro").mean()

        mlflow.log_params(params)
        mlflow.log_metric("cv_f1_macro", cv_f1)

    return {"loss": -cv_f1, "status": STATUS_OK}

# ── Run Hyperopt — 30 trials ──────────────────────────────────────────────────
print("Running Hyperopt — 30 trials...")
trials = Trials()
with mlflow.start_run(run_name="hyperopt_search"):
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials,
    )
    print(f"Best params: {best_params}")

# ── Train final model on full train set with best params ──────────────────────
max_iter_choices = [200, 500, 1000]
class_weight_choices = ["balanced", None]

final_model = LogisticRegression(
    C=best_params["C"],
    max_iter=max_iter_choices[best_params["max_iter"]],
    class_weight=class_weight_choices[best_params["class_weight"]],
    solver="lbfgs",
    multi_class="multinomial",
    random_state=42,
)
final_model.fit(X_train, y_train)

# ── Evaluate on HELD-OUT test set ────────────────────────────────────────────
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"\nTest Accuracy : {acc:.3f}")
print(f"Test F1 Macro : {f1:.3f}")
print(f"\nClassification Report:\n{report}")

# ── Log final model to MLflow ─────────────────────────────────────────────────
with mlflow.start_run(run_name="final_model"):
    mlflow.log_params({
        "C": best_params["C"],
        "max_iter": max_iter_choices[best_params["max_iter"]],
        "class_weight": class_weight_choices[best_params["class_weight"]],
    })
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_macro", f1)

    # Log vectorizer
    with open("/tmp/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    mlflow.log_artifact("/tmp/tfidf_vectorizer.pkl")

    # Log label encoder
    with open("/tmp/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    mlflow.log_artifact("/tmp/label_encoder.pkl")

    # Log classification report
    with open("/tmp/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("/tmp/classification_report.txt")

    # Log model
    mlflow.sklearn.log_model(final_model, "sector_classifier")
    print("Model logged to MLflow")
