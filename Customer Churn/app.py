import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------
# LOAD PKL FILES
# -------------------------
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)   # dict: col -> LabelEncoder

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------
# DEFINE FEATURE CATEGORIES
# ---------------------------------

categorical_columns = list(encoders.keys())  

# NUMERICAL COLUMNS (ONLY 3 SCALED)
scaled_numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# SeniorCitizen is numeric but NOT scaled
unscaled_numeric_columns = ["SeniorCitizen"]

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.write("Provide customer details below to predict churn probability.")

col1, col2 = st.columns(2)

user_inputs = {}

# ---------------------------------
# CATEGORICAL INPUT FIELDS
# ---------------------------------
half = len(categorical_columns) // 2

with col1:
    for col in categorical_columns[:half]:
        options = encoders[col].classes_
        user_inputs[col] = st.selectbox(col, options)

with col2:
    for col in categorical_columns[half:]:
        options = encoders[col].classes_
        user_inputs[col] = st.selectbox(col, options)

# ---------------------------------
# NUMERIC INPUT FIELDS
# ---------------------------------
st.subheader("📌 Numerical Inputs")

colA, colB, colC, colD = st.columns(4)

with colA:
    user_inputs["SeniorCitizen"] = st.selectbox("SeniorCitizen (0 = No, 1 = Yes)", [0, 1])

with colB:
    user_inputs["tenure"] = st.number_input("Tenure", min_value=0, max_value=100, step=1)

with colC:
    user_inputs["MonthlyCharges"] = st.number_input("MonthlyCharges", min_value=0.0)

with colD:
    user_inputs["TotalCharges"] = st.number_input("TotalCharges", min_value=0.0)

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.button("🔍 Predict Churn"):

    final_values = []

    # Encode categorical values
    for col in categorical_columns:
        le = encoders[col]
        final_values.append(le.transform([user_inputs[col]])[0])

    # Add SeniorCitizen (unscaled)
    for col in unscaled_numeric_columns:
        final_values.append(user_inputs[col])

    # Add scaled numerics
    numeric_values = np.array([
        user_inputs["tenure"],
        user_inputs["MonthlyCharges"],
        user_inputs["TotalCharges"]
    ]).reshape(1, -1)

    scaled_values = scaler.transform(numeric_values)[0]

    final_values.extend(list(scaled_values))

    final_data = np.array(final_values).reshape(1, -1)

    # Predict
    prediction = model.predict(final_data)[0]
    probability = model.predict_proba(final_data)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"❗ Customer is likely to CHURN.\n\n**Probability: {probability:.2f}**")
    else:
        st.success(f"✅ Customer will NOT churn.\n\n**Probability: {probability:.2f}**")
