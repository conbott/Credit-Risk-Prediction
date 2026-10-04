# Credit Risk Assessment & Automated Underwriting System

An end-to-end machine learning system designed to predict credit default risk, quantify borrower probability of default, and automate credit underwriting decisions.

---

## 1. Executive Summary

Unmitigated credit default poses substantial financial exposure to commercial and retail lending institutions. Automated credit risk modeling enables financial institutions to evaluate default risk systematically, optimize risk-based pricing, and lower portfolio loss provisions.

This project implements a complete binary classification pipeline that:
- Ingests and processes large-scale credit records (>32,000 observations).
- Employs domain-informed missing value imputation and anomaly filtering.
- Implements a modular `ColumnTransformer` for numerical standardization and categorical encoding.
- Evaluates and compares traditional scorecards (`LogisticRegression`) against sequential boosting architectures (`GradientBoostingClassifier`) using stratified holdout validation.
- Deploys the selected model through an interactive Streamlit underwriting decision application.

---

## 2. Dataset Information

The model was developed on a credit risk dataset comprising 32,581 records (32,410 unique observations following deduplication and anomaly filtering).

### Feature Inventory:
- **Demographics & Employment:** `person_age`, `person_income`, `person_emp_length`, `person_home_ownership` (RENT, OWN, MORTGAGE, OTHER).
- **Loan Attributes:** `loan_intent` (EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT), `loan_grade` (A through G), `loan_amnt`, `loan_int_rate`, `loan_percent_income`.
- **Credit Bureau History:** `cb_person_default_on_file` (Y, N), `cb_person_cred_hist_length`.
- **Target Variable:** `loan_status` (0 = Non-Default, 1 = Default).
- **Baseline Default Rate:** ~21.8% (Imbalanced).

---

## 3. Exploratory Data Analysis Key Findings

1. **Credit Grade Monotonicity:** Default propensity escalates sharply as credit grade declines: Grade A loans exhibit a default rate of ~9.5%, while Grade D reaches ~59% and Grade F/G exceeds ~70%.
2. **Loan-to-Income (LTI) Concentration:** Defaulters exhibit a median loan-to-income ratio of 31%, compared to 13% for non-defaulters. Ratios above 25% signify an elevated risk threshold.
3. **Loan Purpose Variation:** Borrowing driven by acute liquidity distress (`DEBTCONSOLIDATION` and `MEDICAL`) displays default rates of ~27–28%, whereas capital investment loans (`EDUCATION`) exhibit lower default rates (~17%).

---

## 4. Model Training & Evaluation Metrics

Models were evaluated on a held-out stratified 20% test partition (6,482 records) using multi-metric classification criteria:

| Evaluation Metric | Logistic Regression | Gradient Boosting Classifier |
| :--- | :--- | :--- |
| **Train Accuracy** | 86.68% | 94.49% |
| **Test Accuracy** | 86.72% | 93.18% |
| **Precision** | 76.51% | 94.67% |
| **Recall** | 54.46% | 71.85% |
| **F1-Score** | 63.63% | 81.69% |
| **ROC-AUC Score** | 0.8708 | 0.9479 |

### Selection Justification:
The **Gradient Boosting Classifier** was selected for production:
- **Superior Discriminative Performance:** Achieved an ROC-AUC of **0.9479** compared to 0.8708 for Logistic Regression.
- **Enhanced Default Interception:** Captured **71.85% of actual defaults** (Recall), mitigating severe unhedged principal loss.
- **Controlled Variance:** Generalization gap between train and test accuracy remained under 1.4%, confirming negligible overfitting.

---

## 5. Repository Structure

```text
CreditRisk_Project/
├── saved_models/
│   └── credit_risk_pipeline.pkl    # Serialized preprocessor + Gradient Boosting model
├── Credit_risk_pred.ipynb          # End-to-end data pipeline, EDA, and model training
├── credit_app.py                   # Streamlit underwriting application
├── requirements.txt                # Pinned dependency specifications
├── .gitignore                      # Git exclusion rules
└── README.md                       # System documentation
