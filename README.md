# 📊 Customer Churn Prediction & Retention Dashboard

An end-to-end machine learning system and interactive analytics dashboard designed to identify high-risk churn customers, analyze key behavioral churn drivers, and deliver actionable, data-driven retention strategies.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Executive Overview

Customer churn directly erodes recurring revenue and customer lifetime value (LTV). This project shifts churn management from reactive intervention to **proactive retention** by combining:

- **Predictive Modeling:** Supervised ML classifiers (XGBoost, LightGBM, Random Forest) calibrated to prioritize recall and PR-AUC.
- **Model Explainability:** Local and global interpretability using **SHAP** (SHapley Additive exPlanations) values to isolate why individual customers leave.
- **Decision Support System:** A real-time **Streamlit/Dash** interface enabling business and retention teams to simulate interventions, calculate saved revenue, and segment cohorts.

---

## 🎯 Key Features

- **Automated Data Pipeline:** Cleanses raw transactional/demographic data, handles class imbalance via SMOTE/class weighting, and performs feature encoding.
- **Model Benchmarking:** Evaluates multiple algorithms with cross-validation against business-critical metrics (PR-AUC, ROC-AUC, F1-Score, Recall).
- **SHAP-Powered Interpretability:** Explains risk scores at the individual level (waterfall plots) and cohort level (summary beeswarm plots).
- **Interactive ROI & Retention Simulator:** Quantifies potential revenue saved based on targeted discount offers and success rates.
- **Batch & Real-Time Inference:** Allows single-customer profiling via form inputs or bulk CSV batch scoring.

---

## 🧠 Model Architecture & Performance

### Metric Priority
Because false negatives (missing a churning customer) are significantly more costly than false positives (offering a perk to a loyal customer), models are tuned toward **Recall** and **Area Under the Precision-Recall Curve (PR-AUC)**.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.86** | **0.74** | **0.68** | **0.79** | **0.73** |
| Random Forest | 0.84 | 0.70 | 0.65 | 0.74 | 0.69 |
| LightGBM | 0.85 | 0.72 | 0.67 | 0.77 | 0.72 |
| Logistic Regression | 0.78 | 0.58 | 0.54 | 0.68 | 0.60 |

---

## 🛠️ Tech Stack

- **Core & Data Processing:** Python, Pandas, NumPy, Scipy
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn
- **Interpretability:** SHAP
- **Dashboard & UI:** Streamlit / Plotly
- **Artifact Tracking & Serialization:** Joblib / MLflow

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                   # Raw customer dataset (e.g., Telco churn)
│   └── processed/             # Cleaned, transformed feature matrices
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training_and_shap.ipynb
├── models/
│   ├── best_churn_model.pkl   # Serialized pipeline / model weights
│   └── preprocessor.pkl       # Encoders and scaler pipelines
├── src/
│   ├── data_pipeline.py       # Data loading, cleaning, and transformation
│   ├── train.py               # Model training and hyperparameter search
│   └── explainability.py      # SHAP value generation functions
├── app/
│   └── app.py                 # Interactive Streamlit retention dashboard
├── requirements.txt
├── LICENSE
└── README.md

git clone [https://github.com/](https://github.com/)<your-username>/customer-churn-retention.git
cd customer-churn-retention

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
