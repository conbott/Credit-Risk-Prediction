import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Credit Risk Assessment System",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { color: #64748B; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .risk-card { border-radius: 12px; padding: 22px; margin-top: 15px; border: 1px solid #E2E8F0; }
    .high-risk { background-color: #FEF2F2; border-left: 6px solid #DC2626; color: #991B1B; }
    .low-risk { background-color: #ECFDF5; border-left: 6px solid #059669; color: #065F46; }
    .metric-value { font-size: 2.2rem; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Robust Model Loading ---
@st.cache_resource
def load_credit_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "saved_models", "credit_risk_pipeline.pkl")
    
    if not os.path.exists(model_path):
        model_path = "saved_models/credit_risk_pipeline.pkl"
        if not os.path.exists(model_path):
            return None
    return joblib.load(model_path)

pipeline = load_credit_pipeline()

# --- 3. Header ---
st.markdown('<div class="main-title">Credit Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated credit underwriting and default risk classification platform.</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("Model file not found. Ensure `credit_risk_pipeline.pkl` is inside `saved_models/`.")
    st.stop()

# --- 4. Sidebar: Applicant Parameters ---
st.sidebar.header("Applicant and Loan Parameters")

# Demographic & Employment
person_age = st.sidebar.slider("Applicant Age", min_value=18, max_value=90, value=28, step=1)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=5000.0, max_value=500000.0, value=55000.0, step=2500.0)
person_emp_length = st.sidebar.slider("Employment Duration (Years)", min_value=0.0, max_value=45.0, value=4.0, step=0.5)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=0)

# Loan Information
loan_intent = st.sidebar.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], index=0)
loan_grade = st.sidebar.selectbox("Assigned Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=1)
loan_amnt = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500.0, max_value=50000.0, value=10000.0, step=500.0)
loan_int_rate = st.sidebar.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.2, step=0.1)

# Credit Bureau Data
cb_person_default_on_file = st.sidebar.selectbox("Historical Default on File", ["N", "Y"], index=0)
cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (Years)", min_value=1, max_value=30, value=4, step=1)

# Dynamic Loan-to-Income Calculation
loan_percent_income = round(loan_amnt / person_income, 4)

# --- 5. Data Assembly ---
input_df = pd.DataFrame([{
    'person_age': person_age,
    'person_income': person_income,
    'person_home_ownership': person_home_ownership,
    'person_emp_length': person_emp_length,
    'loan_intent': loan_intent,
    'loan_grade': loan_grade,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_default_on_file': cb_person_default_on_file,
    'cb_person_cred_hist_length': cb_person_cred_hist_length
}])

# --- 6. Interface Layout ---
col_profile, col_results = st.columns([1, 1.2], gap="large")

with col_profile:
    st.subheader("Applicant Profile Summary")
    summary_display = input_df.T.rename(columns={0: 'Value'})
    summary_display.loc['loan_percent_income', 'Value'] = f"{loan_percent_income * 100:.2f}%"
    summary_display.loc['person_income', 'Value'] = f"${person_income:,.2f}"
    summary_display.loc['loan_amnt', 'Value'] = f"${loan_amnt:,.2f}"
    summary_display.loc['loan_int_rate', 'Value'] = f"{loan_int_rate:.2f}%"
    st.dataframe(summary_display, use_container_width=True)

with col_results:
    st.subheader("Underwriting Decision")
    
    if st.button("Evaluate Credit Risk", type="primary", use_container_width=True):
        # Model predictions
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1] * 100
        
        # Display Metric & Progress Gauge
        st.metric(label="Estimated Default Probability", value=f"{prob:.1f}%")
        st.progress(min(int(prob), 100))
        
        if pred == 1 or prob >= 50.0:
            st.markdown(f"""
                <div class="risk-card high-risk">
                    <h3>High Risk Tier - Loan Rejection Recommended</h3>
                    <p>The model indicates this applicant carries a substantial likelihood of default.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Primary Underwriting Risk Factors:")
            risk_factors = []
            if loan_percent_income >= 0.25:
                risk_factors.append(f"- **High Debt-to-Income:** Loan represents {loan_percent_income*100:.1f}% of annual salary (benchmark ceiling: 20–25%).")
            if loan_grade in ['D', 'E', 'F', 'G']:
                risk_factors.append(f"- **Adverse Credit Grade:** Assigned grade '{loan_grade}' carries high historical portfolio default propensity.")
            if cb_person_default_on_file == "Y":
                risk_factors.append("- **Derogatory Credit History:** Previous default on file indicates prior credit non-performance.")
            if loan_intent in ['DEBTCONSOLIDATION', 'MEDICAL']:
                risk_factors.append(f"- **High-Risk Loan Purpose:** Stated purpose '{loan_intent}' is associated with cash-flow distress.")
            if not risk_factors:
                risk_factors.append("- Multi-factor risk interaction indicates elevated overall credit loss probability.")
            
            for factor in risk_factors:
                st.write(factor)
                
        else:
            st.markdown(f"""
                <div class="risk-card low-risk">
                    <h3>Low Risk Tier - Loan Approval Recommended</h3>
                    <p>The applicant meets creditworthiness standards with low projected default risk.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Underwriting Highlights:")
            st.write(f"- Manageable debt ratio: Loan represents {loan_percent_income*100:.1f}% of annual income.")
            st.write(f"- Assigned credit tier: Grade {loan_grade}.")
            st.write("- Borrower exhibits adequate capacity to service debt obligations under standard terms.")
