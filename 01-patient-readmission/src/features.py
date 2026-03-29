"""
features.py — sklearn Pipeline for feature engineering
=======================================================
All preprocessing logic lives here. Imported by train.py, predict.py, and tests/.

Design decisions documented in notebooks/02_feature_engineering.ipynb.
Never modify preprocessing logic in notebooks — always modify here.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Columns to drop before any processing
COLS_TO_DROP = [
    'encounter_id',     # identifier
    'patient_nbr',      # identifier
    'weight',           # 96.9% missing — cannot impute
    'payer_code',       # 39.6% missing, not clinically relevant
    'readmitted',       # raw target — replaced by readmitted_30d
]

# Medication columns — 24 drug columns in raw data
MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
]

# Age ordinal order (natural clinical ordering)
AGE_ORDER = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
]

# A1C result ordinal order
A1C_ORDER = ['None', 'Normal', '>7', '>8']

# Glucose serum ordinal order
GLU_ORDER = ['None', 'Normal', '>200', '>300']

# Numeric features to scale
NUMERIC_FEATURES = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses',
    'num_diabetes_meds',   # engineered
]

# Ordinal features with defined ordering
ORDINAL_FEATURES = ['age', 'A1Cresult', 'max_glu_serum']
ORDINAL_CATEGORIES = [AGE_ORDER, A1C_ORDER, GLU_ORDER]

# Nominal categorical features — one-hot encoded
NOMINAL_FEATURES = [
    'race',
    'gender',
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'medical_specialty',
    'diag1_group',
    'diag2_group',
    'diag3_group',
    'change',
    'diabetesMed',
]


# ─────────────────────────────────────────────
# ICD-9 DIAGNOSIS GROUPING
# ─────────────────────────────────────────────

def map_icd9_to_group(code: str) -> str:
    """
    Map raw ICD-9 code to clinical category.

    Groups follow Strack et al. (2014) methodology.
    848 unique diag_1 values → 9 interpretable groups.

    Args:
        code: Raw ICD-9 string from dataset (e.g. '250.83', '648', 'V27')

    Returns:
        Clinical group string
    """
    if pd.isna(code) or code == '?':
        return 'Other'

    code = str(code).strip()

    # Diabetes: 250.xx codes
    if code.startswith('250'):
        return 'Diabetes'

    # E and V codes (external causes, supplemental)
    if code.startswith('E') or code.startswith('V'):
        return 'Other'

    try:
        c = float(code)
        if 390 <= c <= 459 or c == 785:
            return 'Circulatory'
        if 460 <= c <= 519 or c == 786:
            return 'Respiratory'
        if 520 <= c <= 579 or c == 787:
            return 'Digestive'
        if 800 <= c <= 999:
            return 'Injury'
        if 710 <= c <= 739:
            return 'Musculoskeletal'
        if 580 <= c <= 629 or c == 788:
            return 'Genitourinary'
        if 140 <= c <= 239:
            return 'Neoplasms'
    except ValueError:
        pass

    return 'Other'


# ─────────────────────────────────────────────
# FEATURE ENGINEERING TRANSFORMS
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to raw dataframe.

    This function is called BEFORE the sklearn Pipeline.
    The Pipeline handles imputation, encoding, and scaling.

    Args:
        df: Raw dataframe loaded from diabetic_data.csv

    Returns:
        Transformed dataframe ready for sklearn Pipeline
    """
    df = df.copy()

    # 1. Drop irrelevant columns
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Replace '?' with NaN (in case not done at load time)
    df = df.replace('?', np.nan)

    # 3. Deduplicate — keep first encounter per patient
    #    Only applies during training, not inference
    if 'patient_nbr_temp' in df.columns:
        df = df.drop(columns=['patient_nbr_temp'])

    # 4. Map ICD-9 diagnosis codes to clinical groups
    for diag_col, group_col in [('diag_1', 'diag1_group'),
                                  ('diag_2', 'diag2_group'),
                                  ('diag_3', 'diag3_group')]:
        if diag_col in df.columns:
            df[group_col] = df[diag_col].apply(map_icd9_to_group)
            df = df.drop(columns=[diag_col])

    # 5. Insulin flag — binary feature
    #    Insulin is the most commonly prescribed medication (used by ~50% of patients)
    if 'insulin' in df.columns:
        df['on_insulin'] = (df['insulin'] != 'No').astype(int)

    # 6. Count of diabetes medications prescribed (not 'No')
    #    Consolidates 24 medication columns into one interpretable feature
    med_cols_present = [c for c in MED_COLS if c in df.columns]
    df['num_diabetes_meds'] = (df[med_cols_present] != 'No').sum(axis=1)

    # 7. Drop individual medication columns — replaced by num_diabetes_meds + on_insulin
    df = df.drop(columns=[c for c in med_cols_present if c in df.columns])

    # 8. Fill missing values for categorical columns
    #    Done here for columns where 'Unknown' is the correct imputation strategy
    if 'medical_specialty' in df.columns:
        df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')

    # 9. Convert admission_type_id, discharge_disposition_id, admission_source_id to string
    #    These are nominal categories stored as integers — treat as categorical
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Build binary target variable from raw readmitted column.

    Positive class (1): readmitted within 30 days ('<30')
    Negative class (0): readmitted after 30 days ('>30') or not readmitted ('NO')

    Clinical rationale: 30-day threshold is the CMS Hospital Readmissions
    Reduction Program standard. Only early readmissions trigger financial penalties.

    Args:
        df: Raw dataframe containing 'readmitted' column

    Returns:
        Binary Series (0 or 1)
    """
    return (df['readmitted'] == '<30').astype(int)


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only first encounter per patient.

    Multiple encounters from the same patient violate statistical independence —
    required assumption for logistic regression. Consistent with Strack et al. (2014).

    Args:
        df: Raw dataframe with patient_nbr column

    Returns:
        Deduplicated dataframe
    """
    return df.sort_values('encounter_id').drop_duplicates(
        subset='patient_nbr', keep='first'
    ).reset_index(drop=True)


# ─────────────────────────────────────────────
# SKLEARN PIPELINE
# ─────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for the feature pipeline.

    Three parallel pipelines:
    1. Numeric: median imputation → StandardScaler
    2. Ordinal: constant imputation → OrdinalEncoder (preserves order)
    3. Nominal: constant imputation → OneHotEncoder (no order assumed)

    Returns:
        Fitted-ready ColumnTransformer
    """

    # Numeric pipeline: impute missing with median, then scale
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # Ordinal pipeline: fill missing with most frequent, then encode with order
    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])

    # Nominal pipeline: fill missing with 'Unknown', then one-hot encode
    nominal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, NUMERIC_FEATURES),
            ('ordinal', ordinal_pipeline, ORDINAL_FEATURES),
            ('nominal', nominal_pipeline, NOMINAL_FEATURES),
        ],
        remainder='drop',   # drop any columns not explicitly listed
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract feature names after fitting the ColumnTransformer.
    Used for SHAP plots and feature importance charts.

    Args:
        preprocessor: Fitted ColumnTransformer

    Returns:
        List of feature name strings
    """
    feature_names = []

    # Numeric features keep their names
    feature_names.extend(NUMERIC_FEATURES)

    # Ordinal features keep their names
    feature_names.extend(ORDINAL_FEATURES)

    # Nominal features get one-hot expanded names
    ohe = preprocessor.named_transformers_['nominal']['encoder']
    nominal_names = ohe.get_feature_names_out(NOMINAL_FEATURES).tolist()
    feature_names.extend(nominal_names)

    return feature_names