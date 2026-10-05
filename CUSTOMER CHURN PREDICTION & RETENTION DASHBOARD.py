import streamlit as st
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

if sns is not None:
    sns.set_theme(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import io

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if sns is not None:
    sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# 1. Synthetic Data Generation & Model Training Pipeline
# ---------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_samples=1200, random_seed=42):
    """Generates a realistic customer dataset with deliberate churn signals."""
    np.random.seed(random_seed)
    
    tenure_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
    total_charges = np.round(tenure_months * monthly_charges * np.random.uniform(0.95, 1.05, size=n_samples), 2)
    
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_probs = [0.55, 0.25, 0.20]
    contract_type = np.random.choice(contract_choices, size=n_samples, p=contract_probs)
    
    support_tickets = np.random.poisson(lam=1.8, size=n_samples)
    usage_frequency = np.random.randint(1, 35, size=n_samples)
    
    # Calculate underlying churn probability based on realistic business weights
    log_odds = (
        - 0.05 * tenure_months
        + 0.02 * monthly_charges
        + 0.45 * support_tickets
        - 0.08 * usage_frequency
        + np.where(contract_type == "Month-to-month", 1.2, -0.9)
        - 0.8
    )
    prob_churn = 1 / (1 + np.exp(-log_odds))
    churn = np.where(prob_churn > 0.45, 1, 0)
    
    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "support_tickets": support_tickets,
        "usage_frequency": usage_frequency,
        "churn": churn
    })
    return df

@st.cache_resource
def train_pipeline(df):
    """Builds and trains a Random Forest pipeline using ColumnTransformer."""
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    numeric_features = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "usage_frequency"]
    categorical_features = ["contract_type"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42, class_weight="balanced"))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    
    return pipeline, auc_score

# Initialize Data and Model
df_customers = generate_synthetic_data()
pipeline, model_auc = train_pipeline(df_customers)

# ---------------------------------------------------------
# Sidebar Controls & Global Filters
# ---------------------------------------------------------
st.sidebar.title("🎛️ Navigation & Filters")
st.sidebar.markdown(f"**Model Status:** Active  \n**ROC-AUC Score:** `{model_auc:.2f}`")

st.sidebar.subheader("Dataset Quick Filters")
selected_contract = st.sidebar.multiselect(
    "Filter by Contract Type:",
    options=["Month-to-month", "One year", "Two year"],
    default=["Month-to-month", "One year", "Two year"]
)

filtered_df = df_customers[df_customers["contract_type"].isin(selected_contract)]

# ---------------------------------------------------------
# Main Interface: Multi-Tab Layout
# ---------------------------------------------------------
st.title("🛡️ Customer Churn Prediction & Retention Dashboard")
st.caption("Identify high-risk customer turnover, prioritize retention outreach, and score accounts.")

tab1, tab2, tab3 = st.tabs([
    "📈 Executive Overview",
    "🎯 Single Customer Risk Assessor",
    "📁 Batch CSV Scorer"
])

# =========================================================
# TAB 1: Executive Overview
# =========================================================
with tab1:
    st.subheader("Key Business Performance Metrics")
    
    total_cust = len(filtered_df)
    churn_count = filtered_df["churn"].sum()
    churn_rate = (churn_count / total_cust * 100) if total_cust > 0 else 0.0
    
    # Calculate revenue at risk (Monthly charges of churn-flagged customers)
    monthly_rev_at_risk = filtered_df[filtered_df["churn"] == 1]["monthly_charges"].sum()
    total_mrr = filtered_df["monthly_charges"].sum()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Customers", f"{total_cust:,}")
    kpi2.metric("Overall Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate - 20:.1f}% vs Target", delta_color="inverse")
    kpi3.metric("High-Risk Accounts", f"{churn_count:,}")
    kpi4.metric("MRR at Risk", f"${monthly_rev_at_risk:,.2f}", delta=f"{(monthly_rev_at_risk/total_mrr)*100:.1f}% of MRR", delta_color="inverse")
    
    st.divider()
    
    st.subheader("Exploratory Visual Analytics")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Churn Rate by Contract Type**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        contract_churn = filtered_df.groupby("contract_type")["churn"].mean().reset_index()
        contract_churn["churn_percent"] = contract_churn["churn"] * 100
        sns.barplot(data=contract_churn, x="contract_type", y="churn_percent", palette="mako", ax=ax1)
        ax1.set_ylabel("Churn Percentage (%)")
        ax1.set_xlabel("Contract Type")
        ax1.set_ylim(0, 100)
        for p in ax1.patches:
            ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2), ha="center")
        st.pyplot(fig1)
        
    with col_chart2:
        st.markdown("**Churn Distribution by Support Ticket Volume**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x="support_tickets", hue="churn", palette=["#2ecc71", "#e74c3c"], ax=ax2)
        ax2.set_ylabel("Customer Count")
        ax2.set_xlabel("Number of Support Tickets")
        ax2.legend(["Active", "Churned"], title="Status")
        st.pyplot(fig2)

# =========================================================
# TAB 2: Single Customer Risk Assessor
# =========================================================
with tab2:
    st.subheader("Real-Time Customer Scoring & Intervention Engine")
    st.markdown("Enter customer metrics below to predict churn likelihood and receive retention recommendations.")
    
    with st.form("single_customer_form"):
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            input_tenure = st.slider("Account Tenure (Months)", min_value=1, max_value=72, value=12)
            input_contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
            
        with col_input2:
            input_monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=200.0, value=75.0, step=2.5)
            input_total = st.number_input("Total Charges ($)", min_value=15.0, max_value=15000.0, value=input_tenure * input_monthly, step=25.0)
            
        with col_input3:
            input_tickets = st.slider("Recent Support Tickets", min_value=0, max_value=10, value=2)
            input_logins = st.slider("Monthly Logins / Usage Score", min_value=0, max_value=40, value=15)
            
        submit_btn = st.form_submit_button("Assess Churn Risk", use_container_width=True)
        
    if submit_btn:
        customer_row = pd.DataFrame([{
            "tenure_months": input_tenure,
            "monthly_charges": input_monthly,
            "total_charges": input_total,
            "contract_type": input_contract,
            "support_tickets": input_tickets,
            "usage_frequency": input_logins
        }])
        
        churn_probability = pipeline.predict_proba(customer_row)[0][1]
        risk_percent = round(churn_probability * 100, 1)
        
        st.divider()
        st.subheader("Assessment Results")
        
        # Risk gauge meter display
        st.progress(int(risk_percent))
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if risk_percent >= 60.0:
                st.error(f"### High Risk: {risk_percent}%")
                risk_tier = "High"
            elif 30.0 <= risk_percent < 60.0:
                st.warning(f"### Moderate Risk: {risk_percent}%")
                risk_tier = "Moderate"
            else:
                st.success(f"### Low Risk: {risk_percent}%")
                risk_tier = "Low"
                
        with res_col2:
            st.markdown("#### Prescriptive Retention Playbook")
            if risk_tier == "High":
                st.markdown("""
                * **Emergency Outreach:** Assign an Account Manager to schedule a one-on-one resolution call within 24 hours.
                * **Financial Incentive:** Offer an immediate 20% renewal discount for a 1-year contract extension.
                * **Support Escalation:** Review and close any outstanding support tickets with top priority.
                """)
            elif risk_tier == "Moderate":
                st.markdown("""
                * **Proactive Engagement:** Trigger an automated customer satisfaction (CSAT) survey.
                * **Feature Education:** Send an email series containing tutorials for underutilized features.
                * **Check-in Prompt:** Display an in-app prompt offering live assistance.
                """)
            else:
                st.markdown("""
                * **Account Expansion:** Flag account as a prime candidate for tier upgrades and cross-selling.
                * **Advocacy Opportunity:** Request an app store review, case study quote, or referral.
                """)

# =========================================================
# TAB 3: Batch CSV Scorer
# =========================================================
with tab3:
    st.subheader("Bulk Account Scoring & Export")
    st.markdown("Upload a customer CSV file with identical schema columns to score bulk records at once.")
    
    # Downloadable template
    sample_template = df_customers.drop(columns=["churn"]).head(5)
    csv_buffer = io.StringIO()
    sample_template.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download CSV Schema Template",
        data=csv_buffer.getvalue(),
        file_name="customer_churn_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV file for batch processing", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = {"tenure_months", "monthly_charges", "total_charges", "contract_type", "support_tickets", "usage_frequency"}
            
            if not required_cols.issubset(batch_df.columns):
                st.error(f"Missing required columns. Ensure the CSV contains: {', '.join(required_cols)}")
            else:
                batch_probs = pipeline.predict_proba(batch_df)[:, 1]
                batch_preds = (batch_probs >= 0.50).astype(int)
                
                batch_df["churn_probability"] = np.round(batch_probs, 4)
                batch_df["predicted_churn"] = batch_preds
                batch_df["risk_category"] = np.where(
                    batch_df["churn_probability"] >= 0.60, "High Risk",
                    np.where(batch_df["churn_probability"] >= 0.30, "Moderate Risk", "Low Risk")
                )
                
                st.success(f"Successfully processed {len(batch_df)} customer accounts!")
                
                # Summary of batch results
                batch_summary = batch_df["risk_category"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("High-Risk Accounts", batch_summary.get("High Risk", 0))
                c2.metric("Moderate-Risk Accounts", batch_summary.get("Moderate Risk", 0))
                c3.metric("Low-Risk Accounts", batch_summary.get("Low Risk", 0))
                
                st.dataframe(batch_df.sort_values(by="churn_probability", ascending=False), use_container_width=True)
                
                # Download scored batch
                output_csv = io.StringIO()
                batch_df.to_csv(output_csv, index=False)
                
                st.download_button(
                    label="📤 Export Scored Predictions CSV",
                    data=output_csv.getvalue(),
                    file_name="churn_predictions_scored.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"An error occurred while parsing the file: {str(e)}")