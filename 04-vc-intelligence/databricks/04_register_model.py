# ── databricks/04_register_model.py ───────────────────────────────────────────
# Runs in Databricks Community Edition
# Purpose: Register best model from MLflow experiments into Model Registry
# This is the MLOps handoff point — training ends here, serving begins in FastAPI
#
# Interview answer:
# "MLflow logs the model in Databricks. FastAPI loads it via
#  mlflow.pyfunc.load_model() on Cloud Run. Training and serving
#  are intentionally separated — standard MLOps pattern."

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pickle
import pandas as pd

client = MlflowClient()
MODEL_NAME = "vc-sector-classifier"

# ── Find best run from classifier experiment ──────────────────────────────────
# Sort by test_f1_macro descending — pick the best performing run
experiment = client.get_experiment_by_name("/vc-intelligence/classifier-training")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'final_model'",
    order_by=["metrics.test_f1_macro DESC"],
    max_results=1,
)

if not runs:
    raise ValueError("No final_model run found. Run 03_train_classifier.py first.")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_f1 = best_run.data.metrics["test_f1_macro"]
best_acc = best_run.data.metrics["test_accuracy"]

print(f"Best run ID  : {best_run_id}")
print(f"Test F1 Macro: {best_f1:.3f}")
print(f"Test Accuracy: {best_acc:.3f}")

# ── Register model in Model Registry ─────────────────────────────────────────
model_uri = f"runs:/{best_run_id}/sector_classifier"
registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"Registered: {MODEL_NAME} version {registered.version}")

# ── Transition through stages ─────────────────────────────────────────────────
# Staging first — validate before Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered.version,
    stage="Staging",
    archive_existing_versions=False,
)
print(f"Stage: Staging")

# Add description and tags
client.update_model_version(
    name=MODEL_NAME,
    version=registered.version,
    description=(
        f"Sector classifier trained on startup descriptions. "
        f"Test F1 Macro: {best_f1:.3f} | Test Accuracy: {best_acc:.3f}. "
        f"Features: TF-IDF 5000 features, ngram (1,2). "
        f"Trained with Hyperopt 30 trials on Databricks CE."
    ),
)
client.set_model_version_tag(MODEL_NAME, registered.version, "dataset", "startups_sample")
client.set_model_version_tag(MODEL_NAME, registered.version, "feature_type", "tfidf")
client.set_model_version_tag(MODEL_NAME, registered.version, "serving", "fastapi_cloud_run")

# Promote to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=registered.version,
    stage="Production",
    archive_existing_versions=True,  # archives previous Production version
)
print(f"Stage: Production")

# ── Verify export pattern for FastAPI ────────────────────────────────────────
# This is exactly what FastAPI will call on Cloud Run startup
print("\n=== MODEL EXPORT PATTERN FOR FASTAPI ===")
print(f"model_uri = 'models:/{MODEL_NAME}/Production'")
print("model = mlflow.pyfunc.load_model(model_uri)")
print("prediction = model.predict(pd.DataFrame({'text': [description]}))")

# ── Download artifacts for local FastAPI dev ──────────────────────────────────
# FastAPI also needs the vectorizer and label encoder
artifacts = client.list_artifacts(best_run_id)
print(f"\nArtifacts in best run:")
for a in artifacts:
    print(f"  {a.path}")

print("\nTo load in FastAPI:")
print("  mlflow.artifacts.download_artifacts(")
print(f"      run_id='{best_run_id}',")
print("      artifact_path='tfidf_vectorizer.pkl'")
print("  )")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== REGISTRY SUMMARY ===")
mv = client.get_model_version(MODEL_NAME, registered.version)
print(f"Name        : {mv.name}")
print(f"Version     : {mv.version}")
print(f"Stage       : {mv.current_stage}")
print(f"Run ID      : {mv.run_id}")
print(f"Description : {mv.description}")
