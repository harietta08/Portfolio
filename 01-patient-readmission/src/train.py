"""
train.py — Training script with MLflow logging and model registration
=====================================================================
Trains 3 models: Logistic Regression, Random Forest, XGBoost
Logs all params, metrics, and artifacts to MLflow
Registers best model to MLflow Model Registry

Run with: python src/train.py
Or:        make train
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (
    engineer_features,
    build_target,
    deduplicate_patients,
    build_preprocessor,
    get_feature_names,
)

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH = os.getenv('DATA_PATH', 'data/raw/diabetic_data.csv')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'mlflow/mlruns')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'patient-readmission')
MODEL_NAME = os.getenv('MODEL_NAME', 'readmission-classifier')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cost matrix — clinical cost of each error type
# False Negative: missed high-risk patient → readmitted → $15,000 cost + patient harm
# False Positive: unnecessary intervention → ~$500 follow-up call
COST_FALSE_NEGATIVE = 15000
COST_FALSE_POSITIVE = 500


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute full clinical metrics at a given decision threshold.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        threshold: Decision threshold (default 0.5, optimized later)

    Returns:
        Dictionary of metric name → value
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Clinical cost calculation
    total_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)

    return {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'avg_precision': average_precision_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold,
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
        'total_clinical_cost': int(total_cost),
        'cost_per_patient': round(total_cost / len(y_true), 2),
    }


def optimize_threshold(y_true, y_pred_proba):
    """
    Find decision threshold that minimizes total clinical cost.

    At default 0.5, many high-risk patients are missed (false negatives).
    Lowering the threshold catches more high-risk patients at the cost
    of more false positives — but false positives are much cheaper clinically.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Optimal threshold float
    """
    best_threshold = 0.5
    best_cost = float('inf')

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return round(best_threshold, 2)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_and_prepare_data():
    """
    Load raw data, deduplicate, engineer features, build target.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    print("Loading data...")
    raw = pd.read_csv(DATA_PATH, na_values='?')
    print(f"  Raw shape: {raw.shape}")

    # Build target before deduplication (needs readmitted column)
    raw = deduplicate_patients(raw)
    print(f"  After deduplication: {raw.shape}")

    y = build_target(raw)
    X = engineer_features(raw)

    print(f"  Positive class: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Feature shape after engineering: {X.shape}")

    # Train/test split — stratified to preserve class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit preprocessor on training data only
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    # Apply SMOTE to training data only
    # SMOTE oversamples minority class synthetically
    # Applied AFTER train/test split to prevent data leakage
    print(f"  Applying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
    print(f"  After SMOTE: {X_train_proc.shape}, positive rate: {y_train.mean()*100:.1f}%")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_models():
    """
    Return the three models with their parameters and rationale.

    Logistic Regression: interpretable baseline
    Random Forest: ensemble, handles non-linearity
    XGBoost: boosted, typically best performance on tabular data
    """
    return {
        'logistic_regression': {
            'model': LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                C=1.0,
                solver='lbfgs',
            ),
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'solver': 'lbfgs',
            },
            'rationale': 'Interpretable baseline. Coefficients map directly to clinical risk factors.',
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_leaf': 20,
                'class_weight': 'balanced',
            },
            'rationale': 'Handles non-linear relationships (e.g. age-risk interaction). More robust than single trees.',
        },
        'xgboost': {
            'model': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=8,  # handles class imbalance (ratio ~1:8)
                random_state=RANDOM_STATE,
                eval_metric='aucpr',
                verbosity=0,
            ),
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'scale_pos_weight': 8,
            },
            'rationale': 'Best performance on tabular clinical data. scale_pos_weight handles imbalance.',
        },
    }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train_and_log(model_name, model_config, X_train, X_test, y_train, y_test,
                  preprocessor, feature_names):
    """
    Train one model, log everything to MLflow, return metrics.

    Args:
        model_name: String identifier
        model_config: Dict with model, params, rationale
        X_train/X_test: Processed feature arrays
        y_train/y_test: Target arrays
        preprocessor: Fitted ColumnTransformer
        feature_names: List of feature name strings

    Returns:
        Dict of metrics for comparison
    """
    model = model_config['model']
    params = model_config['params']

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('model_type', model_name)
        mlflow.log_param('training_samples', len(X_train))
        mlflow.log_param('test_samples', len(X_test))
        mlflow.log_param('rationale', model_config['rationale'])

        # Train
        print(f"\n  Training {model_name}...")
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Optimize threshold for clinical cost
        optimal_threshold = optimize_threshold(y_test, y_pred_proba)
        print(f"  Optimal threshold: {optimal_threshold}")

        # Compute metrics at optimal threshold
        metrics = compute_metrics(y_test, y_pred_proba, threshold=optimal_threshold)

        # Log all metrics
        mlflow.log_metrics({
            'auc_roc': metrics['auc_roc'],
            'avg_precision': metrics['avg_precision'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'total_clinical_cost': metrics['total_clinical_cost'],
            'cost_per_patient': metrics['cost_per_patient'],
            'optimal_threshold': optimal_threshold,
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'true_negatives': metrics['tn'],
            'false_negatives': metrics['fn'],
        })

        # Log model with signature
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            model,
            artifact_path='model',
            signature=signature,
            registered_model_name=f"{MODEL_NAME}-{model_name}",
        )

        # Save threshold and feature names as artifacts
        artifacts = {
            'optimal_threshold': optimal_threshold,
            'feature_names': feature_names,
            'model_name': model_name,
            'cost_false_negative': COST_FALSE_NEGATIVE,
            'cost_false_positive': COST_FALSE_POSITIVE,
        }
        import tempfile, os
        tmp_path = os.path.join(tempfile.gettempdir(), 'model_artifacts.json')
        with open(tmp_path, 'w') as f:
            json.dump(artifacts, f)
        mlflow.log_artifact(tmp_path)

        run_id = mlflow.active_run().info.run_id
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Recall:  {metrics['recall']:.4f}")
        print(f"  Clinical cost: ${metrics['total_clinical_cost']:,}")
        print(f"  MLflow run ID: {run_id}")

        metrics['run_id'] = run_id
        metrics['model_name'] = model_name
        return metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("=" * 60)
    print("Patient Readmission Prediction — Model Training")
    print("=" * 60)

    # Load and prepare data
    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_prepare_data()

    # Train all three models
    models = get_models()
    all_results = []

    print("\nTraining 3 models with MLflow tracking...")
    for model_name, model_config in models.items():
        metrics = train_and_log(
            model_name, model_config,
            X_train, X_test, y_train, y_test,
            preprocessor, feature_names,
        )
        all_results.append(metrics)

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'AUC-ROC':<10} {'Recall':<10} {'F1':<10} {'Clinical Cost':<15}")
    print("-" * 70)

    best_model = None
    best_cost = float('inf')

    for r in all_results:
        print(f"{r['model_name']:<25} {r['auc_roc']:<10.4f} {r['recall']:<10.4f} "
              f"{r['f1']:<10.4f} ${r['total_clinical_cost']:<14,}")
        if r['total_clinical_cost'] < best_cost:
            best_cost = r['total_clinical_cost']
            best_model = r

    print("\n" + "=" * 60)
    print(f"WINNER: {best_model['model_name']}")
    print(f"  AUC-ROC: {best_model['auc_roc']:.4f}")
    print(f"  Recall:  {best_model['recall']:.4f} (catches {best_model['recall']*100:.1f}% of high-risk patients)")
    print(f"  Clinical cost: ${best_model['total_clinical_cost']:,}")
    print(f"  False negatives (missed patients): {best_model['fn']}")
    print(f"  MLflow run ID: {best_model['run_id']}")
    print("=" * 60)

    # Save best model info for API to load
    best_model_info = {
        'model_name': best_model['model_name'],
        'run_id': best_model['run_id'],
        'optimal_threshold': best_model['threshold'],
        'auc_roc': best_model['auc_roc'],
        'recall': best_model['recall'],
    }
    os.makedirs('mlflow', exist_ok=True)
    with open('mlflow/best_model.json', 'w') as f:
        json.dump(best_model_info, f, indent=2)
    print(f"\nBest model info saved to mlflow/best_model.json")
    print("Run `mlflow ui` to view all experiments in the browser.")


if __name__ == '__main__':
    main()
