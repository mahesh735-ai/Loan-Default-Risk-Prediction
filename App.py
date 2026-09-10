import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="FinRisk Premier — Underwriting Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM BANKING UI CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #F0F4F8 0%, #E2E8F0 100%);
        color: #1E293B;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 2px solid #38BDF8;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    .header-box {
        background: linear-gradient(90deg, #0284C7 0%, #0D9488 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.30);
    }
    .header-box h1 { margin: 0; font-size: 28px; font-weight: 700; color: #FFFFFF !important; }
    .header-box p { color: #FFFFFF !important; margin-bottom: 0; }
    .approved-box {
        background-color: #ECFDF5; border: 2px solid #10B981;
        padding: 20px; border-radius: 12px; color: #065F46;
        text-align: center; margin-top: 15px;
    }
    .approved-box h3, .approved-box p { color: #065F46 !important; }
    .rejected-box {
        background-color: #FEF2F2; border: 2px solid #EF4444;
        padding: 20px; border-radius: 12px; color: #991B1B;
        text-align: center; margin-top: 15px;
    }
    .rejected-box h3, .rejected-box p { color: #991B1B !important; }
    div.stButton > button {
        background: linear-gradient(90deg, #0284C7 0%, #0D9488 100%);
        color: white !important; font-weight: 600; font-size: 16px;
        border: none; border-radius: 8px; padding: 12px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.40);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_xgb_model.json"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

# =========================================================
# SAFE MODEL & METADATA LOADING
# =========================================================
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found:\n{METADATA_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    metadata = joblib.load(METADATA_PATH)

    if "feature_names" not in metadata:
        raise KeyError("feature_names not found in model_metadata.pkl")
    if "optimal_threshold" not in metadata:
        raise KeyError("optimal_threshold not found in model_metadata.pkl")

    feature_names = list(metadata["feature_names"])
    optimal_threshold = float(metadata["optimal_threshold"])

    if len(feature_names) == 0:
        raise ValueError("Feature names list is empty.")

    return {"model": model, "feature_names": feature_names, "optimal_threshold": optimal_threshold}

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    feature_names = artifacts["feature_names"]
    optimal_threshold = artifacts["optimal_threshold"]
except Exception as e:
    st.error("🚨 Unable to Load the Machine Learning Model")
    st.code(str(e))
    st.warning("""
Make sure these two files are in the SAME folder as app.py:

1. credit_risk_xgb_model.json
2. model_metadata.pkl
""")
    st.stop()

# =========================================================
# TOP HEADER
# =========================================================
st.markdown("""
<div class="header-box">
    <h1>🏦 FinRisk Premier — Underwriting Decision Engine</h1>
    <p style="margin-top: 8px; opacity: 0.9;">
        Automated Subprime Loan Default Assessment & Risk Scoring System
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR — APPLICANT FINANCIAL PROFILE
# =========================================================
st.sidebar.header("📋 Applicant Financial Profile")

person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=28, step=1)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, value=65000, step=1000)
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.sidebar.selectbox("Assigned Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500, value=12000, step=500)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=11.5, step=0.1)
cb_person_default_on_file = st.sidebar.selectbox("Past Default History", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, max_value=40, value=6, step=1)

# =========================================================
# FEATURE CALCULATIONS
# =========================================================
loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
annual_interest_amount = loan_amnt * (loan_int_rate / 100)
loan_per_emp_year = loan_amnt / (person_emp_length + 1)
cred_hist_age_ratio = cb_person_cred_hist_length / person_age if person_age > 0 else 0

# =========================================================
# PREPARE INPUT DICTIONARY
# =========================================================
input_dict = {feature: 0 for feature in feature_names}

def set_feature(feature_name, value):
    if feature_name in input_dict:
        input_dict[feature_name] = value

set_feature("person_age", person_age)
set_feature("person_income", person_income)
set_feature("person_emp_length", person_emp_length)
set_feature("loan_amnt", loan_amnt)
set_feature("loan_int_rate", loan_int_rate)
set_feature("loan_percent_income", loan_percent_income)
set_feature("cb_person_cred_hist_length", cb_person_cred_hist_length)
set_feature("annual_interest_amount", annual_interest_amount)
set_feature("loan_per_emp_year", loan_per_emp_year)
set_feature("cred_hist_age_ratio", cred_hist_age_ratio)

grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
set_feature("loan_grade_encoded", grade_map[loan_grade])

set_feature(f"person_home_ownership_{person_home_ownership}", 1)
set_feature(f"loan_intent_{loan_intent}", 1)
set_feature("cb_person_default_on_file_encoded", 1 if cb_person_default_on_file == "Y" else 0)

if person_income <= 38500:
    set_feature("income_group_Low_Income", 1)
elif person_income <= 55000:
    set_feature("income_group_Medium_Income", 1)
elif person_income <= 79200:
    set_feature("income_group_High_Income", 1)
else:
    set_feature("income_group_Very_High_Income", 1)

# =========================================================
# FINAL DATAFRAME
# =========================================================
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_names, fill_value=0)
input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

# =========================================================
# MAIN DASHBOARD LAYOUT
# =========================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.container(border=True):
        st.subheader("💡 Calculated Risk Metrics")
        st.write(f"**Debt-to-Income Ratio:** `{loan_percent_income:.2%}`")
        st.write(f"**Annual Interest Expense:** `${annual_interest_amount:,.2f}`")
        st.write(f"**Credit/Age Exposure Ratio:** `{cred_hist_age_ratio:.2f}`")
        st.write(f"**Loan per Employment Year:** `${loan_per_emp_year:,.2f}`")

with col2:
    with st.container(border=True):
        st.subheader("🎯 Automated Assessment Engine")
        st.write("Click below to run real-time risk decisioning using the trained XGBoost model.")
        st.caption(f"Decision Threshold: {optimal_threshold:.2f}")

        if st.button("Evaluate Application", type="primary"):
            try:
                prediction_proba = model.predict_proba(input_df)
                prob_default = float(prediction_proba[0][1])

                if not (0 <= prob_default <= 1):
                    raise ValueError("Invalid probability returned by model.")

                st.markdown("---")

                # ⭐ FIXED: High-contrast, dynamically colored probability display
                risk_color = "#DC2626" if prob_default >= optimal_threshold else "#059669"

                st.markdown(f"""
<div style="text-align:center; margin-bottom: 15px;">
    <p style="font-size:16px; color:#475569; margin-bottom:4px; font-weight:600;">
        Estimated Default Probability
    </p>
    <p style="font-size:52px; font-weight:800; color:{risk_color}; margin:0; line-height:1.1;">
        {prob_default * 100:.2f}%
    </p>
</div>
""", unsafe_allow_html=True)

                if prob_default >= optimal_threshold:
                    st.markdown(f"""
<div class="rejected-box">
    <h3>🚨 HIGH RISK DETECTED</h3>
    <p><b>Decision: REJECTED / MANUAL UNDERWRITING REQUIRED</b></p>
    <p>Default Probability exceeds the safety threshold of {optimal_threshold:.2f}.</p>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="approved-box">
    <h3>✅ APPROVED</h3>
    <p><b>Decision: AUTOMATED DISBURSEMENT READY</b></p>
    <p>Default Probability is within acceptable risk limits (&lt; {optimal_threshold:.2f}).</p>
</div>
""", unsafe_allow_html=True)

            except Exception as prediction_error:
                st.error("🚨 Prediction could not be completed.")
                st.code(str(prediction_error))

st.markdown("---")
st.caption("FinRisk Premier | XGBoost-Based Loan Default Risk Assessment System")