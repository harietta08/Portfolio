"""
export_model.py — Export trained model for HuggingFace deployment
Run once locally: python export_model.py
Creates: hf_deploy/model_bundle.pkl
"""

import os
import sys
import pickle
import json
sys.path.append('.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.features import (
    engineer_features, build_target, deduplicate_patients,
    build_preprocessor, get_feature_names
)
from src.train import get_models, optimize_threshold

print("Loading data...")
raw = pd.read_csv('data/raw/diabetic_data.csv', na_values='?')
raw = deduplicate_patients(raw)
y = build_target(raw)
X = engineer_features(raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Fitting preprocessor...")
preprocessor = build_preprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
feature_names = get_feature_names(preprocessor)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

print("Training XGBoost...")
model = get_models()['xgboost']['model']
model.fit(X_train_bal, y_train_bal)

y_proba = model.predict_proba(X_test_proc)[:, 1]
threshold = optimize_threshold(y_test, y_proba)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}, Threshold: {threshold}")

# Bundle everything needed for inference
bundle = {
    'model': model,
    'preprocessor': preprocessor,
    'feature_names': feature_names,
    'optimal_threshold': threshold,
    'auc_roc': auc,
    'model_name': 'xgboost',
}

os.makedirs('hf_deploy', exist_ok=True)
with open('hf_deploy/model_bundle.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print(f"Model bundle saved to hf_deploy/model_bundle.pkl")
print(f"Bundle size: {os.path.getsize('hf_deploy/model_bundle.pkl') / 1024 / 1024:.1f} MB")