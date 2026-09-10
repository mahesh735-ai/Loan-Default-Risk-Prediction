# Loan-Default-Risk-Prediction
⚡End-to-end Loan Default Risk Prediction using Python, XGBoost &amp; SHAP. Includes data cleaning, EDA, feature engineering, model comparison, threshold tuning and explainable ML, achieving 94.94% ROC-AUC and 86% default recall. Deployed with Streamlit.

# FinRisk Premier — Loan Default Risk Prediction

An end-to-end machine learning classification system that predicts the probability of loan default, enabling automated, consistent, and risk-adjusted lending decisions.

---

## Motivation

This project is personally motivated. Growing up, I would accompany my father to the bank during loan applications and observed how lending decisions were made based on income stability, repayment history, and asset value. This project revisits that same decision-making process — but through data and machine learning, from the perspective of a Data Analyst rather than a customer.

---

## Problem Statement

A lending institution wants to predict, at the time of loan application, whether an applicant is likely to default — using their demographic profile, income, employment stability, and existing credit history — enabling faster, consistent, risk-adjusted approval decisions instead of manual underwriting.

**Target Variable:** `loan_status` — 1 = default/high-risk, 0 = repaid/healthy
**Class Distribution:** ~78% healthy, ~22% default (moderately imbalanced)

---

## Dataset

**Source:** [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
32,581 loan applications with 11 features covering demographics, income, employment, loan terms, and credit bureau history.

---

## Project Workflow

**1. Data Cleaning**
- Removed 165 duplicate records and unrealistic outliers (`person_age` > 100, `person_emp_length` > 60 years).
- Applied median imputation for `person_emp_length`.
- Applied **grade-wise median imputation** for `loan_int_rate` — since interest rates vary systematically by risk grade (A–G), imputing by group rather than a single overall median preserves this relationship.

**2. Exploratory Data Analysis**
- Identified strong multicollinearity between `person_age` and `cb_person_cred_hist_length` (r = 0.86).
- Found `cb_person_default_on_file` (prior default history) to be a strong univariate predictor of default rate.
- Analyzed default rate by loan grade, home ownership, and loan intent.

**3. Feature Engineering**
- Ordinal encoding for `loan_grade` (A–G, order-preserving).
- Engineered interaction features: `annual_interest_amount`, `loan_per_emp_year`, `cred_hist_age_ratio`.
- Income quartile bucketing (`income_group`).
- One-hot encoding for nominal categorical variables.

**4. Model Selection**
- Compared Logistic Regression, Random Forest, and XGBoost.
- Used **different feature sets per model family**: multicollinear features were dropped for the linear model (Logistic Regression) but retained for tree-based models, which are not affected by correlated inputs in the same way.
- Applied `class_weight` / `scale_pos_weight` to address class imbalance rather than oversampling, to preserve the natural data distribution.

| Model | ROC-AUC | Recall (Default) | Precision (Default) |
|---|---|---|---|
| Logistic Regression | 0.8663 | 0.62 | 0.70 |
| Random Forest | 0.9356 | 0.72 | 0.85 |
| **XGBoost (Final)** | **0.9494** | **0.81** | **0.83** |

**5. Hyperparameter Tuning (Validated Separately)**
`RandomizedSearchCV` was run to validate the manually-configured baseline. The tuned model showed a negligible improvement (ROC-AUC 0.9498 vs 0.9494), confirming the manually-set `scale_pos_weight` (3.5, derived from the actual class imbalance ratio of ~3.58) was already near-optimal. The simpler baseline model was retained for production. Full experiment: **[LINK]**

**6. Explainability & Threshold Tuning**
- Used **SHAP** (TreeExplainer) to interpret global feature importance and individual predictions.
- Tuned the decision threshold from the default 0.50 to **0.35** using the Precision-Recall curve — improving defaulter recall from 81% to 86%, reflecting that missing a genuine defaulter is costlier to the lender than a false alarm on a safe applicant.

**7. Deployment**
- Model serialized in XGBoost's native JSON format (version-safe, avoids pickle corruption issues) alongside a metadata file storing the feature schema and optimal threshold.
- Deployed as an interactive Streamlit web application for real-time risk assessment.

---

## Key Results

- **ROC-AUC:** 0.9494
- **Recall (Default class, at tuned threshold):** 86%
- **Top Risk Drivers (SHAP):** Loan grade, income, debt-to-income ratio, prior default history

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Deployment | Streamlit |

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/loan-default-risk-prediction.git
cd loan-default-risk-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open the local URL shown in the terminal, enter applicant details in the sidebar, and click "Evaluate Application."

---

## Project Structure

```
loan-default-risk-prediction/
├── app.py                              # Streamlit deployment app
├── Loan_Default_Risk_Prediction.ipynb  # Full ML pipeline notebook
├── credit_risk_xgb_model.json          # Serialized trained XGBoost model
├── model_metadata.pkl                  # Feature schema + decision threshold
├── requirements.txt
├── experiments/
│   └── hyperparameter_tuning_experiment.ipynb
└── README.md
```

---

## Key Learnings

- Feature scaling and multicollinearity handling should be tailored to the model family, not applied uniformly — linear models are sensitive to correlated inputs and unscaled features; tree-based ensembles are not.
- A model's cross-validation score is not the final word — held-out test set performance is what determines production readiness.
- The default 0.50 classification threshold is rarely optimal in imbalanced, cost-asymmetric business problems; threshold selection should be driven by the relative cost of false positives vs. false negatives.
- Systematic hyperparameter search does not always outperform a well-reasoned manual configuration — validating this, rather than assuming tuning always helps, is itself a useful exercise.
