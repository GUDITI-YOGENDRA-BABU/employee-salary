import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model_pipeline = joblib.load('salary_prediction_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'salary_prediction_model.pkl' not found. "
             "Please ensure you've trained and saved your model first by running your training script.")
    st.stop()

# --- Define options for categorical features based on your data ---
# REPLACE THESE LISTS WITH THE ACTUAL UNIQUE VALUES FROM YOUR PROCESSED DATA
# (After filtering, but BEFORE Label Encoding in your original script).
# Use data['column_name'].unique().tolist() from your training script to get accurate lists.
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'NOT LISTED',
                     'State-gov', 'Federal-gov', 'Self-emp-inc']
marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced',
                          'Separated', 'Widowed', 'Married-spouse-absent']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical',
                      'Sales', 'Other-service', 'Tech-support', 'Transport-moving',
                      'Farming-fishing', 'Machine-op-inspct', 'Protective-serv', 'others']
relationship_options = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                          'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica',
                          'China', 'Columbia', 'Italy', 'Dominican-Republic', 'South', 'Japan',
                          'Guatemala', 'Vietnam', 'Ecuador', 'Poland', 'Iran', 'Taiwan',
                          'Portugal', 'Peru', 'France', 'Cambodia', 'Greece', 'Nicaragua',
                          'Ireland', 'Trinadad&Tobago', 'Thailand', 'Outlying-US(Guam-USVI-etc)',
                          'Hong', 'Haiti', 'NOT COUNTABLE']

# This function simulates the Label Encoding from your training script
# It's crucial this mapping is consistent with your training data's encoding
def encode_categorical_input(df):
    encoded_df = df.copy()
    
    # It's safer to use a consistent encoder. Since LabelEncoder assigns based on sort order
    # of unique values, we recreate that here.
    # A more robust solution would be to save the fitted LabelEncoders.
    
    # Map options to their expected integer labels based on your training script's LabelEncoder behavior
    # Assuming `LabelEncoder` sorts categories alphabetically for integer assignment.
    # You might need to verify the exact integer mapping by inspecting your training script's output
    # or by saving the LabelEncoder objects.

    le_mappings = {
        'workclass': {val: i for i, val in enumerate(sorted(workclass_options))},
        'marital-status': {val: i for i, val in enumerate(sorted(marital_status_options))},
        'occupation': {val: i for i, val in enumerate(sorted(occupation_options))},
        'relationship': {val: i for i, val in enumerate(sorted(relationship_options))},
        'race': {val: i for i, val in enumerate(sorted(race_options))},
        'gender': {val: i for i, val in enumerate(sorted(gender_options))},
        'native-country': {val: i for i, val in enumerate(sorted(native_country_options))}
    }

    for col, mapping in le_mappings.items():
        encoded_df[col] = encoded_df[col].map(mapping)
        # Handle potential unseen categories by mapping to a default (e.g., -1 or 0, depending on model)
        # Or raise an error. For now, we'll map to NaN and then handle.
        if encoded_df[col].isnull().any():
            st.warning(f"Unseen category detected in '{col}'. This might affect prediction.")
            encoded_df[col] = encoded_df[col].fillna(0) # Default to 0 for unseen

    # Apply the additional numerical filters from your training script AFTER encoding
    # This is unusual for a live app input, as it implies input should already be within these ranges.
    # Consider if these filters are truly necessary for new input or were just for training data cleaning.
    # For robust apps, you'd rather raise an error for out-of-range encoded values.
    # For now, we'll include them as per your training script.
    encoded_df = encoded_df[(encoded_df['workclass'] <=5)&(encoded_df['workclass'] >=1)]
    encoded_df = encoded_df[(encoded_df['marital-status'] <=5)&(encoded_df['marital-status'] >=1)]
    encoded_df = encoded_df[(encoded_df['native-country'] <=35)&(encoded_df['native-country'] >=5)]

    return encoded_df


st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.title("💰 Employee Salary Prediction")
st.markdown("---")

st.header("Employee Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 70, 35)
    workclass = st.selectbox("Workclass", workclass_options)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=200000)

with col2:
    education_num = st.slider("Education Num", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", marital_status_options)
    occupation = st.selectbox("Occupation", occupation_options)

with col3:
    relationship = st.selectbox("Relationship", relationship_options)
    race = st.selectbox("Race", race_options)
    gender = st.selectbox("Gender", gender_options)

st.markdown("---")
st.header("Financial & Work Details")
col4, col5 = st.columns(2)

with col4:
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

with col5:
    hours_per_week = st.slider("Hours Per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", native_country_options)

# Create DataFrame from user inputs
input_data = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education-num': education_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}])

# Encode categorical features for the input data
processed_input = encode_categorical_input(input_data.copy())

# Define the exact feature order as used during training (after dropping 'income')
# This must match the order of 'x' in your training script after all preprocessing
feature_order = [
    'age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
]

# Ensure all features are present and in the correct order,
# fill missing (e.g., due to filtering out encoded values) with 0 or a suitable default
processed_input = processed_input.reindex(columns=feature_order, fill_value=0)


st.markdown("---")
if st.button("Predict Salary Bracket", type="primary"):
    if processed_input.empty:
        st.error("Invalid input after internal filtering. Please adjust inputs.")
    else:
        try:
            prediction = model_pipeline.predict(processed_input)
            prediction_proba = model_pipeline.predict_proba(processed_input)

            st.subheader("Prediction Result:")
            if prediction[0] == '>50K':
                st.success(f"## Predicted Income: **>50K** 🚀")
                st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
            else:
                st.info(f"## Predicted Income: **<=50K** 👇")
                st.write(f"Confidence: **{prediction_proba[0][0]*100:.2f}%**")

            st.markdown("---")
            st.write("**:bulb: Disclaimer:** This prediction is based on a machine learning model trained on historical data and should be used for informational purposes only. It does not guarantee future outcomes.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check your input values. If the issue persists, contact support.")
