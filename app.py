from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ==================================================
# STREAMLIT APP
# ==================================================

st.set_page_config(
    page_title="Cirrhosis Stage Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Cirrhosis Stage Prediction")
st.write("Enter the patient details below to predict the cirrhosis stage.")

# ==================================================
# FILE PATHS
# ==================================================

DATA_PATH = Path("data/cirrhosis.csv")
MODEL_PATH = Path("models/cirrhosis_flask_pipeline.pkl")

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_PATH)

# ==================================================
# FEATURES USED BY THE MODEL
# ID and Stage are excluded
# ==================================================

feature_columns = [
    "N_Days",
    "Status",
    "Drug",
    "Age",
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin"
]

# ==================================================
# NUMERICAL FEATURES
# ==================================================

numeric_features = [
    "N_Days",
    "Age",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin"
]

# ==================================================
# CATEGORICAL FEATURES
# ==================================================

categorical_features = [
    "Status",
    "Drug",
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema"
]

# ==================================================
# DROPDOWN VALUES
# ==================================================

category_values = {}

for col in categorical_features:
    category_values[col] = (
        df[col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

# ==================================================
# IQR BOUNDS USED DURING TRAINING
# ==================================================

iqr_bounds = {
    "Bilirubin": (-3.1375, 7.3625),
    "Cholesterol": (23.75, 625.75),
    "Copper": (-81.375, 245.625),
    "Alk_Phos": (-791.25, 3642.75),
    "Tryglicerides": (-15.875, 251.125),
    "Platelets": (15.625, 506.625),
    "Albumin": (2.575, 4.535),
    "SGOT": (-26.35, 258.85),
    "Prothrombin": (8.35, 12.75),
    "Age": (7496.875, 28645.875)
}

# ==================================================
# LOAD SAVED MODEL
# ==================================================

model = joblib.load(MODEL_PATH)

# ==================================================
# INPUT SECTION
# ==================================================

st.header("Patient Information")

input_data = {}

col1, col2 = st.columns(2)

# ==================================================
# CATEGORICAL INPUTS
# ==================================================

with col1:

    st.subheader("Categorical Features")

    for feature in categorical_features:

        input_data[feature] = st.selectbox(
            feature,
            category_values[feature]
        )

# ==================================================
# NUMERICAL INPUTS
# ==================================================

with col2:

    st.subheader("Numerical Features")

    for feature in numeric_features:

        input_data[feature] = st.number_input(
            feature,
            value=0.0
        )

# ==================================================
# PREDICTION BUTTON
# ==================================================

if st.button("🔍 Predict Cirrhosis Stage"):

    try:

        # ------------------------------------------
        # CREATE DATAFRAME
        # ------------------------------------------

        row = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # ------------------------------------------
        # CONVERT NUMERICAL VALUES
        # ------------------------------------------

        for col in numeric_features:

            row[col] = pd.to_numeric(
                row[col],
                errors="raise"
            )

        # ------------------------------------------
        # APPLY IQR CAPPING
        # SAME AS TRAINING DATA
        # ------------------------------------------

        for col, (lower, upper) in iqr_bounds.items():

            row[col] = row[col].clip(
                lower=lower,
                upper=upper
            )

        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        prediction = model.predict(row)[0]

        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        st.success(
            f"Predicted Cirrhosis Stage: {prediction}"
        )

        # ------------------------------------------
        # PREDICTION PROBABILITIES
        # ------------------------------------------

        if hasattr(model, "predict_proba"):

            probability_values = model.predict_proba(row)[0]
            classes = model.classes_

            probabilities = {
                str(label): round(
                    float(probability) * 100,
                    2
                )
                for label, probability
                in zip(classes, probability_values)
            }

            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame(
                list(probabilities.items()),
                columns=["Stage", "Probability (%)"]
            )

            st.dataframe(
                probability_df,
                use_container_width=True
            )

            st.bar_chart(
                probability_df.set_index("Stage")
            )

    except Exception as e:

        st.error(f"Error: {e}")