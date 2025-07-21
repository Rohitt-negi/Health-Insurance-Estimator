# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Cache the ML model as a global, shared resource
@st.cache_resource
def load_model():
    return joblib.load('health_insurance_cost_estimator.pkl')

# Cache income mapping data
@st.cache_data
def get_income_mapping():
    return {
        '<10L': 5,
        '10L-25L': 17.5,
        '25L-40L': 32.5,
        '>40L': 50
    }

model = load_model()
income_mapping = get_income_mapping()

st.title("💡 Health Insurance Cost Estimator")

st.sidebar.header("User Input Parameters")
age = st.sidebar.slider("Age", 18, 100, 45)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
region = st.sidebar.selectbox("Region", ['North', 'South', 'East', 'West', 'Northeast'])
marital_status = st.sidebar.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Widowed'])
dependants = st.sidebar.number_input("Number of Dependents", 0, 10, 0)
bmi_cat = st.sidebar.selectbox("BMI Category", ['Underweight', 'Normal', 'Overweight', 'Obese'])
smoking = st.sidebar.selectbox("Smoking Status", ['Smoking', 'No Smoking'])
employment = st.sidebar.selectbox("Employment Status", ['Salaried', 'Self‑employed', 'Unemployed', 'Retired'])
income_level = st.sidebar.selectbox("Income Level", list(income_mapping.keys()))
medical_history = st.sidebar.text_input("Medical History (comma-separated or 'No Disease')", 'No Disease')
insurance_plan = st.sidebar.selectbox("Insurance Plan", ['Bronze', 'Silver', 'Gold', 'Platinum'])

# Prepare single-row input DataFrame
num_med_cond = 0 if medical_history.strip().lower() == 'no disease' else len(medical_history.split(','))
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Region': [region],
    'Marital_status': [marital_status],
    'Number_Of_Dependants': [dependants],
    'BMI_Category': [bmi_cat],
    'Smoking_Status': [smoking],
    'Employment_Status': [employment],
    'Medical_History': [medical_history],
    'Insurance_Plan': [insurance_plan],
    'Num_Medical_Conditions': [num_med_cond],
    'Income_Numerical': [income_mapping[income_level]]
})

if st.sidebar.button("Estimate Premium"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Annual Premium: ₹{prediction:,.2f}")

    # Show top 10 feature importances if available
    try:
        proc = model.named_steps['preprocessor']
        mod = model.named_steps['model']
        num_feats = proc.transformers_[0][2]
        cat_feats = proc.transformers_[1][1] \
            .named_steps['onehot'] \
            .get_feature_names_out(proc.transformers_[1][2])
        feat_names = np.concatenate([num_feats, cat_feats])
        importances = mod.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': feat_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)

        st.subheader("Top 10 Feature Importances")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax)
        st.pyplot(fig)
    except Exception:
        st.info("Feature importances not available for this model.")

st.markdown("---")
st.markdown("#### How to use")
st.markdown("""
- Use the sidebar to set your inputs.
- Click **Estimate Premium** to see the predicted annual cost.
- If supported, the chart shows the top 10 features influencing the prediction.
""")
