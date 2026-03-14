import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------

st.set_page_config(
    page_title="Fintech Fraud Detection System",
    layout="wide"
)

st.title("💳 Fintech Fraud Detection System")
st.markdown("AI-powered credit card fraud detection using Machine Learning.")

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------

model = joblib.load("fraud_model_final.pkl")
threshold = joblib.load("threshold.pkl")

# ---------------------------------------------
# SIDEBAR INFORMATION
# ---------------------------------------------

st.sidebar.header("📊 Model Information")

st.sidebar.success(f"Fraud Threshold: {threshold}")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.metric("Model Accuracy", "99.9%")
st.sidebar.metric("Fraud Recall", "83%")
st.sidebar.metric("ROC-AUC Score", "0.965")

st.sidebar.markdown("---")

st.sidebar.subheader("🎓 Student Information")

st.sidebar.write("**Full Name:** Adeoti Toheeb Oluwaseun")
st.sidebar.write("**Matric No:** AI/HND/F24/0093")
st.sidebar.write("**Project Topic:**")
st.sidebar.write("Fraud Detection in Financial Transactions for Fintech Firms Using Random Forest")

st.sidebar.markdown("---")

st.sidebar.info("Random Forest Fraud Detection Model")

# ---------------------------------------------
# EXPECTED FEATURE ORDER
# ---------------------------------------------

feature_columns = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

# ---------------------------------------------
# MANUAL TRANSACTION INPUT
# ---------------------------------------------

with st.expander("🔎 Manual Transaction Check (Click to Open)"):

    st.write("Enter transaction features manually to check fraud probability.")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input("Transaction Time", value=0.0)

    with col2:
        amount = st.number_input("Transaction Amount", value=100.0)

    st.subheader("PCA Features (V1 - V28)")

    v_inputs = []

    cols = st.columns(4)

    for i in range(1, 29):
        col = cols[(i-1) % 4]
        with col:
            v = st.number_input(f"V{i}", value=0.0)
            v_inputs.append(v)

    if st.button("Check Transaction"):

        features = [time] + v_inputs + [amount]

        input_df = pd.DataFrame([features], columns=feature_columns)

        probability = model.predict_proba(input_df)[0][1]

        prediction = 1 if probability >= threshold else 0

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        st.write(f"Fraud Probability: **{probability:.6f}**")

        # Gauge Chart

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Fraud Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# BATCH FRAUD DETECTION
# ---------------------------------------------

st.header("📂 Batch Fraud Detection")

uploaded_file = st.file_uploader(
    "Upload CSV file containing transactions",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    if st.button("Run Fraud Detection"):

        X = df.drop(columns=["Class"], errors="ignore")

        X = X[feature_columns]

        probabilities = model.predict_proba(X)[:,1]

        predictions = (probabilities >= threshold).astype(int)

        df["Fraud_Probability"] = probabilities
        df["Prediction"] = predictions

        st.success("Fraud Detection Completed")

        # ---------------------------------------------
        # SUMMARY METRICS
        # ---------------------------------------------

        total_transactions = len(df)
        fraud_transactions = df["Prediction"].sum()
        fraud_rate = (fraud_transactions / total_transactions) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total_transactions)
        col2.metric("Fraud Detected", fraud_transactions)
        col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")

        # ---------------------------------------------
        # FRAUD BAR CHART
        # ---------------------------------------------

        st.subheader("Fraud vs Legitimate Transactions")

        chart_data = df["Prediction"].value_counts().reset_index()

        chart_data.columns = ["Transaction Type", "Count"]

        chart_data["Transaction Type"] = chart_data["Transaction Type"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        fig = px.bar(
            chart_data,
            x="Transaction Type",
            y="Count",
            title="Fraud Detection Results"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------
        # FRAUD PROBABILITY DISTRIBUTION
        # ---------------------------------------------

        st.subheader("Fraud Probability Distribution")

        fig2 = px.histogram(
            df,
            x="Fraud_Probability",
            nbins=50,
            title="Distribution of Fraud Probability Scores"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ---------------------------------------------
        # FRAUDULENT TRANSACTIONS
        # ---------------------------------------------

        st.subheader("🚨 Detected Fraudulent Transactions")

        fraud_df = df[df["Prediction"] == 1]

        if len(fraud_df) > 0:

            st.write(f"Total Fraudulent Transactions Found: **{len(fraud_df)}**")

            fraud_df = fraud_df.sort_values(
                by="Fraud_Probability",
                ascending=False
            )

            st.dataframe(
                fraud_df.style.background_gradient(
                    subset=["Fraud_Probability"],
                    cmap="Reds"
                )
            )

        else:

            st.success("No fraudulent transactions detected in this dataset.")

        # ---------------------------------------------
        # TOP HIGH RISK TRANSACTIONS
        # ---------------------------------------------

        st.subheader("🔥 Top High-Risk Transactions")

        top_risk = df.sort_values(
            by="Fraud_Probability",
            ascending=False
        ).head(10)

        st.dataframe(top_risk)

        # ---------------------------------------------
        # DOWNLOAD FRAUD REPORT
        # ---------------------------------------------

        fraud_csv = fraud_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Fraud Transactions",
            data=fraud_csv,
            file_name="fraud_transactions_report.csv",
            mime="text/csv"
        )

        # ---------------------------------------------
        # DOWNLOAD FULL RESULTS
        # ---------------------------------------------

        full_csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Full Detection Results",
            data=full_csv,
            file_name="full_fraud_detection_results.csv",
            mime="text/csv"
        )