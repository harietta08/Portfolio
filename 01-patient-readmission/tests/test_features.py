"""
test_features.py — Unit tests for src/features.py
===================================================
Run with: pytest tests/test_features.py -v
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from src.features import (
    map_icd9_to_group,
    engineer_features,
    build_target,
    deduplicate_patients,
    build_preprocessor,
    get_feature_names,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal raw dataframe mimicking diabetic_data.csv structure."""
    return pd.DataFrame({
        'encounter_id':            [1, 2, 3, 4],
        'patient_nbr':             [101, 102, 101, 103],  # patient 101 appears twice
        'race':                    ['Caucasian', 'AfricanAmerican', None, 'Hispanic'],
        'gender':                  ['Female', 'Male', 'Female', 'Male'],
        'age':                     ['[70-80)', '[50-60)', '[60-70)', '[40-50)'],
        'weight':                  [None, None, None, None],
        'admission_type_id':       [1, 2, 1, 3],
        'discharge_disposition_id':[1, 1, 2, 1],
        'admission_source_id':     [7, 7, 1, 7],
        'time_in_hospital':        [5, 3, 7, 2],
        'payer_code':              [None, None, None, None],
        'medical_specialty':       ['Cardiology', None, 'InternalMedicine', None],
        'num_lab_procedures':      [44, 31, 55, 20],
        'num_procedures':          [1, 0, 2, 0],
        'num_medications':         [14, 8, 20, 5],
        'number_outpatient':       [0, 1, 0, 2],
        'number_emergency':        [0, 0, 1, 0],
        'number_inpatient':        [2, 0, 3, 0],
        'diag_1':                  ['410', '250.83', '486', '820'],
        'diag_2':                  ['250', '401', '250', '250'],
        'diag_3':                  ['401', None, '428', '401'],
        'number_diagnoses':        [9, 5, 8, 4],
        'max_glu_serum':           ['None', 'None', '>200', 'None'],
        'A1Cresult':               ['None', '>8', 'None', 'Normal'],
        'metformin':               ['No', 'Steady', 'No', 'Up'],
        'repaglinide':             ['No', 'No', 'No', 'No'],
        'nateglinide':             ['No', 'No', 'No', 'No'],
        'chlorpropamide':          ['No', 'No', 'No', 'No'],
        'glimepiride':             ['No', 'No', 'No', 'No'],
        'acetohexamide':           ['No', 'No', 'No', 'No'],
        'glipizide':               ['No', 'No', 'Steady', 'No'],
        'glyburide':               ['No', 'No', 'No', 'No'],
        'tolbutamide':             ['No', 'No', 'No', 'No'],
        'pioglitazone':            ['No', 'No', 'No', 'No'],
        'rosiglitazone':           ['No', 'No', 'No', 'No'],
        'acarbose':                ['No', 'No', 'No', 'No'],
        'miglitol':                ['No', 'No', 'No', 'No'],
        'troglitazone':            ['No', 'No', 'No', 'No'],
        'tolazamide':              ['No', 'No', 'No', 'No'],
        'examide':                 ['No', 'No', 'No', 'No'],
        'citoglipton':             ['No', 'No', 'No', 'No'],
        'insulin':                 ['Steady', 'Up', 'No', 'Steady'],
        'glyburide-metformin':     ['No', 'No', 'No', 'No'],
        'glipizide-metformin':     ['No', 'No', 'No', 'No'],
        'glimepiride-pioglitazone':['No', 'No', 'No', 'No'],
        'metformin-rosiglitazone': ['No', 'No', 'No', 'No'],
        'metformin-pioglitazone':  ['No', 'No', 'No', 'No'],
        'change':                  ['Ch', 'No', 'Ch', 'No'],
        'diabetesMed':             ['Yes', 'Yes', 'Yes', 'No'],
        'readmitted':              ['<30', 'NO', '>30', '<30'],
    })


# ─────────────────────────────────────────────
# ICD-9 MAPPING TESTS
# ─────────────────────────────────────────────

class TestICD9Mapping:

    def test_diabetes_code(self):
        assert map_icd9_to_group('250.83') == 'Diabetes'
        assert map_icd9_to_group('250') == 'Diabetes'

    def test_circulatory_code(self):
        assert map_icd9_to_group('410') == 'Circulatory'
        assert map_icd9_to_group('440') == 'Circulatory'

    def test_respiratory_code(self):
        assert map_icd9_to_group('486') == 'Respiratory'

    def test_injury_code(self):
        assert map_icd9_to_group('820') == 'Injury'

    def test_v_code_returns_other(self):
        assert map_icd9_to_group('V27') == 'Other'

    def test_e_code_returns_other(self):
        assert map_icd9_to_group('E11') == 'Other'

    def test_none_returns_other(self):
        assert map_icd9_to_group(None) == 'Other'

    def test_nan_returns_other(self):
        import numpy as np
        assert map_icd9_to_group(np.nan) == 'Other'

    def test_unknown_code_returns_other(self):
        assert map_icd9_to_group('999999') == 'Other'


# ─────────────────────────────────────────────
# TARGET BUILDING TESTS
# ─────────────────────────────────────────────

class TestBuildTarget:

    def test_less_than_30_is_positive(self, sample_raw_df):
        y = build_target(sample_raw_df)
        assert y.iloc[0] == 1  # '<30' → 1

    def test_no_readmission_is_negative(self, sample_raw_df):
        y = build_target(sample_raw_df)
        assert y.iloc[1] == 0  # 'NO' → 0

    def test_greater_than_30_is_negative(self, sample_raw_df):
        y = build_target(sample_raw_df)
        assert y.iloc[2] == 0  # '>30' → 0

    def test_output_is_binary(self, sample_raw_df):
        y = build_target(sample_raw_df)
        assert set(y.unique()).issubset({0, 1})

    def test_output_length_matches_input(self, sample_raw_df):
        y = build_target(sample_raw_df)
        assert len(y) == len(sample_raw_df)


# ─────────────────────────────────────────────
# DEDUPLICATION TESTS
# ─────────────────────────────────────────────

class TestDeduplication:

    def test_removes_duplicate_patients(self, sample_raw_df):
        deduped = deduplicate_patients(sample_raw_df)
        assert deduped['patient_nbr'].nunique() == len(deduped)

    def test_keeps_first_encounter(self, sample_raw_df):
        deduped = deduplicate_patients(sample_raw_df)
        # patient 101 has encounter_id 1 and 3 — should keep 1
        patient_101 = deduped[deduped['patient_nbr'] == 101]
        assert patient_101['encounter_id'].values[0] == 1

    def test_output_has_fewer_rows(self, sample_raw_df):
        deduped = deduplicate_patients(sample_raw_df)
        assert len(deduped) < len(sample_raw_df)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING TESTS
# ─────────────────────────────────────────────

class TestEngineerFeatures:

    def test_drops_weight_column(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'weight' not in X.columns

    def test_drops_payer_code(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'payer_code' not in X.columns

    def test_drops_encounter_id(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'encounter_id' not in X.columns

    def test_creates_diag_groups(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'diag1_group' in X.columns
        assert 'diag2_group' in X.columns
        assert 'diag3_group' in X.columns

    def test_creates_on_insulin_flag(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'on_insulin' in X.columns
        assert set(X['on_insulin'].unique()).issubset({0, 1})

    def test_creates_num_diabetes_meds(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'num_diabetes_meds' in X.columns
        assert X['num_diabetes_meds'].min() >= 0

    def test_fills_missing_race(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert X['race'].isna().sum() == 0

    def test_fills_missing_medical_specialty(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert X['medical_specialty'].isna().sum() == 0

    def test_no_raw_medication_columns_remain(self, sample_raw_df):
        X = engineer_features(sample_raw_df)
        assert 'metformin' not in X.columns
        assert 'insulin' not in X.columns


# ─────────────────────────────────────────────
# PIPELINE TESTS
# ─────────────────────────────────────────────

class TestPipeline:

    def test_preprocessor_produces_no_nans(self, sample_raw_df):
        deduped = deduplicate_patients(sample_raw_df)
        X = engineer_features(deduped)
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        assert not np.isnan(X_proc).any()

    def test_output_is_numeric(self, sample_raw_df):
        deduped = deduplicate_patients(sample_raw_df)
        X = engineer_features(deduped)
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        assert X_proc.dtype in [np.float32, np.float64]

    def test_feature_names_match_output_columns(self, sample_raw_df):
        deduped = deduplicate_patients(sample_raw_df)
        X = engineer_features(deduped)
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
        feature_names = get_feature_names(preprocessor)
        assert len(feature_names) == X_proc.shape[1]
