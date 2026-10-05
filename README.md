# 📊 Customer Churn Prediction & Retention Dashboard

An interactive Streamlit dashboard that predicts customer churn and turns risk scores into actionable retention steps.

🔗 **Live Demo:** https://customer-churn-prediction-retention-dashboard-3gxujhncuuzkdd4n.streamlit.app/

> **Note:** This project runs on a **synthetic dataset** generated in code, built to demonstrate the end-to-end ML workflow.

## ✨ Features

- **📈 Executive Overview:** Total customers, churn rate, high-risk accounts, and MRR at risk, with contract-type filters and charts.
- **🎯 Single Customer Risk Assessor:** Enter a customer's details to get a churn probability (Low / Moderate / High) and a retention playbook.
- **📁 Batch CSV Scorer:** Upload a CSV, score many accounts at once, and export the results. A schema template is included.

## 🧠 Model

- **Algorithm:** Random Forest Classifier (`n_estimators=120`, `max_depth=6`, `class_weight="balanced"`)
- **Pipeline:** `ColumnTransformer` (One-Hot Encoding for contract type) + classifier in a single scikit-learn `Pipeline`
- **Evaluation:** 80/20 stratified train-test split, ROC-AUC shown in the app sidebar
- **Risk tiers:** High (≥ 60%), Moderate (30-60%), Low (< 30%)

## 🛠️ Tech Stack

Python · Streamlit · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## 🚀 Run Locally

```bash
git clone https://github.com/harshkumarjaiswal7/CUSTOMER-CHURN-PREDICTION-RETENTION-DASHBOARD.git
cd CUSTOMER-CHURN-PREDICTION-RETENTION-DASHBOARD
pip install -r requirements.txt
streamlit run "CUSTOMER CHURN PREDICTION & RETENTION DASHBOARD.py"
```

## 🔮 Future Improvements

- Use a real dataset (e.g., Telco Customer Churn)
- Add SHAP explainability
- Compare more models (XGBoost, LightGBM)
- Add an ROI / retention-offer simulator

## 👤 Author

**Harsh Kumar Jaiswal**
[LinkedIn](YOUR_LINKEDIN_URL) · [GitHub](https://github.com/harshkumarjaiswal7)
