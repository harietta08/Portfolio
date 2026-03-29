"""
schemas.py — Pydantic request and response models
==================================================
Defines the contract between API callers and the model.
Invalid requests are rejected before they reach the model.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS — valid values for categorical fields
# ─────────────────────────────────────────────

class AgeGroup(str, Enum):
    age_0_10   = "[0-10)"
    age_10_20  = "[10-20)"
    age_20_30  = "[20-30)"
    age_30_40  = "[30-40)"
    age_40_50  = "[40-50)"
    age_50_60  = "[50-60)"
    age_60_70  = "[60-70)"
    age_70_80  = "[70-80)"
    age_80_90  = "[80-90)"
    age_90_100 = "[90-100)"


class A1CResult(str, Enum):
    none   = "None"
    normal = "Normal"
    high_7 = ">7"
    high_8 = ">8"


class GlucoseResult(str, Enum):
    none    = "None"
    normal  = "Normal"
    high200 = ">200"
    high300 = ">300"


class MedicationStatus(str, Enum):
    no     = "No"
    steady = "Steady"
    up     = "Up"
    down   = "Down"


# ─────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Patient data required for readmission risk prediction.
    All fields map directly to features used in training.
    """

    # Demographics
    race: Optional[str] = Field(default="Unknown", description="Patient race")
    gender: str = Field(..., description="Patient gender: Male or Female")
    age: AgeGroup = Field(..., description="Age group in 10-year intervals")

    # Admission info
    admission_type_id: int = Field(..., ge=1, le=8, description="Admission type (1-8)")
    discharge_disposition_id: int = Field(..., ge=1, le=28, description="Discharge disposition (1-28)")
    admission_source_id: int = Field(..., ge=1, le=25, description="Admission source (1-25)")

    # Clinical measurements
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital (1-14)")
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab procedures")
    num_procedures: int = Field(..., ge=0, description="Number of non-lab procedures")
    num_medications: int = Field(..., ge=0, description="Number of distinct medications")
    number_diagnoses: int = Field(..., ge=0, description="Number of diagnoses")

    # Prior visits
    number_outpatient: int = Field(..., ge=0, description="Outpatient visits in prior year")
    number_emergency: int = Field(..., ge=0, description="Emergency visits in prior year")
    number_inpatient: int = Field(..., ge=0, description="Inpatient visits in prior year")

    # Lab results
    max_glu_serum: GlucoseResult = Field(default=GlucoseResult.none, description="Glucose serum result")
    A1Cresult: A1CResult = Field(default=A1CResult.none, description="HbA1c test result")

    # Diagnoses (ICD-9 codes)
    diag_1: Optional[str] = Field(default=None, description="Primary diagnosis ICD-9 code")
    diag_2: Optional[str] = Field(default=None, description="Secondary diagnosis ICD-9 code")
    diag_3: Optional[str] = Field(default=None, description="Additional diagnosis ICD-9 code")

    # Medications
    insulin: MedicationStatus = Field(default=MedicationStatus.no, description="Insulin status")
    medical_specialty: Optional[str] = Field(default="Unknown", description="Admitting physician specialty")
    change: str = Field(default="No", description="Change in diabetic medications: Ch or No")
    diabetesMed: str = Field(default="No", description="Any diabetes medication prescribed: Yes or No")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[70-80)",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "time_in_hospital": 5,
                "num_lab_procedures": 44,
                "num_procedures": 1,
                "num_medications": 14,
                "number_diagnoses": 9,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 2,
                "max_glu_serum": "None",
                "A1Cresult": "None",
                "diag_1": "410",
                "diag_2": "250",
                "diag_3": "401",
                "insulin": "Steady",
                "medical_specialty": "Cardiology",
                "change": "Ch",
                "diabetesMed": "Yes"
            }
        }
    }


# ─────────────────────────────────────────────
# RESPONSE SCHEMA
# ─────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Prediction result with risk score and top contributing factors.
    """
    readmission_probability: float = Field(
        ..., description="Probability of 30-day readmission (0.0 to 1.0)"
    )
    risk_level: str = Field(
        ..., description="Risk category: LOW, MEDIUM, or HIGH"
    )
    decision_threshold: float = Field(
        ..., description="Threshold used for HIGH/LOW classification"
    )
    flagged_for_intervention: bool = Field(
        ..., description="True if patient should receive proactive intervention"
    )
    top_risk_factors: list = Field(
        default=[], description="Top features driving this prediction"
    )
    model_version: str = Field(
        ..., description="Model name and version used for prediction"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "readmission_probability": 0.73,
                "risk_level": "HIGH",
                "decision_threshold": 0.14,
                "flagged_for_intervention": True,
                "top_risk_factors": [
                    {"feature": "number_inpatient", "impact": "high", "direction": "increases_risk"},
                    {"feature": "time_in_hospital", "impact": "medium", "direction": "increases_risk"}
                ],
                "model_version": "xgboost-v1"
            }
        }
    }


class HealthResponse(BaseModel):
    """
    API health check response.
    Used by load balancers, monitoring systems, and deployment pipelines.
    """
    status: str = Field(..., description="API status: healthy or degraded")
    model_name: str = Field(..., description="Name of loaded model")
    model_version: str = Field(..., description="Version of loaded model")
    optimal_threshold: float = Field(..., description="Decision threshold in use")
    auc_roc: float = Field(..., description="Model AUC-ROC on test set")
