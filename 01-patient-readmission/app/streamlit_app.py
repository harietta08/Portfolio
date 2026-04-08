"""
streamlit_app.py — Patient Readmission Risk Prediction UI
==========================================================
Deployed to: HuggingFace Spaces (permanent free URL)

Run locally: streamlit run app/streamlit_app.py
Or:          make app
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Patient Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD MODEL DIRECTLY (no API dependency)
# ─────────────────────────────────────────────

@st.cache_resource
def load_model_and_preprocessor():
    """Load model once and cache it for all sessions."""
    import mlflow
    from src.predict import load_model
    model, preprocessor, model_info = load_model()
    return model, preprocessor, model_info


@st.cache_resource
def load_explainer(_model):
    """Load SHAP explainer once and cache it."""
    import shap
    return shap.TreeExplainer(_model)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏥 Patient Readmission Risk Prediction")
st.markdown(
    """
    **Clinical Decision Support System** — Predicts 30-day hospital readmission risk
    for diabetic patients at time of discharge.

    > Each prevented readmission saves approximately **$15,000** in hospital costs
    and reduces patient harm. This tool helps clinical staff prioritize
    follow-up interventions before discharge.
    """
)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — PATIENT INPUT FORM
# ─────────────────────────────────────────────

st.sidebar.header("Patient Information")
st.sidebar.markdown("Enter patient details at time of discharge.")

with st.sidebar:
    # Demographics
    st.subheader("Demographics")
    race = st.selectbox("Race", [
        "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"
    ])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.selectbox("Age Group", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ], index=6)

    # Admission Details
    st.subheader("Admission Details")
    admission_type_id = st.selectbox(
        "Admission Type",
        options=[1, 2, 3, 4, 5, 6, 7, 8],
        format_func=lambda x: {
            1: "Emergency", 2: "Urgent", 3: "Elective",
            4: "Newborn", 5: "Not Available", 6: "NULL",
            7: "Trauma Center", 8: "Not Mapped"
        }.get(x, str(x))
    )
    discharge_disposition_id = st.slider("Discharge Disposition ID", 1, 28, 1)
    admission_source_id = st.slider("Admission Source ID", 1, 25, 7)
    time_in_hospital = st.slider("Days in Hospital", 1, 14, 4)
    medical_specialty = st.selectbox("Admitting Physician Specialty", [
        "Unknown", "Cardiology", "InternalMedicine", "Family/GeneralPractice",
        "Surgery-General", "Orthopedics", "Gastroenterology", "Pulmonology",
        "Nephrology", "Hematology/Oncology", "Other"
    ])

    # Clinical Measurements
    st.subheader("Clinical Measurements")
    num_lab_procedures = st.slider("Number of Lab Procedures", 0, 120, 44)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    num_medications = st.slider("Number of Medications", 0, 80, 14)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 9)
    max_glu_serum = st.selectbox("Glucose Serum Result", ["None", "Normal", ">200", ">300"])
    A1Cresult = st.selectbox("HbA1c Result", ["None", "Normal", ">7", ">8"])

    # Prior Visits
    st.subheader("Prior Year Visits")
    number_outpatient = st.slider("Outpatient Visits", 0, 40, 0)
    number_emergency = st.slider("Emergency Visits", 0, 40, 0)
    number_inpatient = st.slider("Inpatient Visits", 0, 20, 0)

    # Diagnoses
    st.subheader("Diagnoses (ICD-9)")
    diag_1 = st.text_input("Primary Diagnosis", value="410")
    diag_2 = st.text_input("Secondary Diagnosis", value="250")
    diag_3 = st.text_input("Additional Diagnosis", value="401")

    # Medications
    st.subheader("Medications")
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    change = st.selectbox("Medication Change", ["No", "Ch"])
    diabetesMed = st.selectbox("Diabetes Medication Prescribed", ["Yes", "No"])

    predict_button = st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN PANEL — RESULTS
# ─────────────────────────────────────────────

if not predict_button:
    # Default state
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "XGBoost", "Best clinical cost")
    col2.metric("AUC-ROC", "0.652", "Test set")
    col3.metric("Recall", "99.1%", "High-risk patients caught")

    st.info(
        "👈 Fill in patient details in the sidebar and click "
        "**Predict Readmission Risk** to generate a prediction."
    )

    st.subheader("About This Model")
    st.markdown("""
    This clinical decision support system was trained on **70,000 real patient encounters**
    from 130 US hospitals (1999–2008). It predicts which patients are at high risk of
    being readmitted within 30 days of discharge.

    **Why this matters:**
    - Hospital readmissions cost the US healthcare system ~$26 billion annually
    - The CMS Hospital Readmissions Reduction Program financially penalizes hospitals
      with high readmission rates
    - Early identification allows clinical staff to intervene before discharge

    **How to use:**
    1. Enter patient details in the left sidebar
    2. Click Predict Readmission Risk
    3. Review the risk score and SHAP explanation
    4. Use the explanation to inform discharge planning decisions

    **Important:** This tool is for decision support only. Clinical judgment should
    always take precedence.
    """)

else:
    # Build patient dict
    patient_data = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "diag_1": diag_1 if diag_1 else None,
        "diag_2": diag_2 if diag_2 else None,
        "diag_3": diag_3 if diag_3 else None,
        "insulin": insulin,
        "medical_specialty": medical_specialty,
        "change": change,
        "diabetesMed": diabetesMed,
    }

    with st.spinner("Running prediction..."):
        try:
            from src.predict import predict_single, load_model, _prepare_single_patient
            from src.features import build_preprocessor, get_feature_names, engineer_features
            import shap
            import pandas as pd

            model, preprocessor, model_info = load_model_and_preprocessor()
            threshold = model_info.get("optimal_threshold", 0.5)

            result = predict_single(patient_data)
            prob = result["readmission_probability"]
            risk_level = result["risk_level"]
            flagged = result["flagged_for_intervention"]

            # ── RISK SCORE DISPLAY ──
            st.subheader("Prediction Result")
            col1, col2, col3 = st.columns(3)

            risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            risk_emoji = risk_colors.get(risk_level, "⚪")

            col1.metric(
                label="Readmission Probability",
                value=f"{prob:.1%}",
                delta=f"Threshold: {threshold:.0%}",
                delta_color="off"
            )
            col2.metric(
                label="Risk Level",
                value=f"{risk_emoji} {risk_level}",
            )
            col3.metric(
                label="Intervention Recommended",
                value="Yes ⚠️" if flagged else "No ✅",
            )

            # Risk gauge
            fig, ax = plt.subplots(figsize=(8, 1.5))
            ax.barh(0, 1, color='#f0f0f0', height=0.5)
            bar_color = '#e74c3c' if prob >= 0.6 else '#f39c12' if prob >= 0.3 else '#2ecc71'
            ax.barh(0, prob, color=bar_color, height=0.5)
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.0%})')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Readmission Probability')
            ax.set_title('Risk Score', fontweight='bold')
            ax.legend(loc='upper right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            st.pyplot(fig)
            plt.close()

            # Clinical cost framing
            if flagged:
                st.error(
                    f"⚠️ **HIGH RISK PATIENT** — Proactive intervention recommended.\n\n"
                    f"Estimated cost if readmitted: **$15,000**. "
                    f"A follow-up call or enhanced discharge planning costs ~$500."
                )
            else:
                st.success(
                    f"✅ **LOWER RISK PATIENT** — Standard discharge protocol appropriate.\n\n"
                    f"Predicted readmission probability: {prob:.1%} "
                    f"(below {threshold:.0%} intervention threshold)."
                )

            st.divider()

            # ── SHAP EXPLANATION ──
            st.subheader("Why This Prediction? — SHAP Explanation")
            st.markdown(
                "The chart below shows which patient factors drove this risk score. "
                "**Red bars** push risk higher. **Blue bars** push risk lower."
            )

            with st.spinner("Computing SHAP explanation..."):
                try:
                    explainer = load_explainer(model)
                    df_patient = pd.DataFrame([patient_data])
                    df_eng = _prepare_single_patient(df_patient)
                    X_proc = preprocessor.transform(df_eng)
                    shap_vals = explainer(X_proc)
                    feature_names = get_feature_names(preprocessor)

                    # Manual waterfall plot (more control than shap.plots.waterfall)
                    sv = shap_vals[0].values
                    top_idx = np.argsort(np.abs(sv))[::-1][:12]
                    top_names = [feature_names[i] for i in top_idx]
                    top_vals = [sv[i] for i in top_idx]

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    colors = ['#e74c3c' if v > 0 else '#3498db' for v in top_vals]
                    ax2.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
                    ax2.set_yticks(range(len(top_names)))
                    ax2.set_yticklabels(top_names[::-1], fontsize=9)
                    ax2.axvline(x=0, color='black', linewidth=0.8)
                    ax2.set_xlabel('SHAP Value (impact on prediction)')
                    ax2.set_title(
                        f'Feature Impact — Patient Risk: {prob:.1%}',
                        fontweight='bold'
                    )
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    st.pyplot(fig2)
                    plt.close()

                    # Top factors as text
                    st.markdown("**Top risk factors for this patient:**")
                    for name, val in zip(top_names[:5], top_vals[:5]):
                        direction = "increases" if val > 0 else "decreases"
                        arrow = "↑" if val > 0 else "↓"
                        st.markdown(f"- `{name}` {arrow} {direction} readmission risk")

                except Exception as shap_err:
                    st.warning(f"SHAP explanation unavailable: {shap_err}")

            st.divider()

            # ── MODEL INFO ──
            with st.expander("Model Details"):
                st.json({
                    "model": model_info.get("model_name"),
                    "auc_roc": model_info.get("auc_roc"),
                    "optimal_threshold": threshold,
                    "cost_false_negative": "$15,000",
                    "cost_false_positive": "$500",
                    "training_data": "70,000 diabetic encounters, 130 US hospitals, 1999-2008",
                })

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Make sure the API is running: `uvicorn api.main:app --port 8080`")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Built by Hari Etta | MS Data Science & AI, Illinois Institute of Technology |
    <a href='https://github.com/harietta08/Portfolio'>GitHub</a>
    <br>
    ⚠️ For demonstration purposes only. Not for clinical use without formal validation.
    </div>
    """,
    unsafe_allow_html=True
)