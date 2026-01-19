import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Visa Processing Time Estimator",
    layout="centered"
)

HISTORY_FILE = "prediction_history.csv"

# =========================
# PERSISTENT STORAGE
# =========================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=[
        "Country",
        "Visa Type",
        "Application Month",
        "Age",
        "Travel History",
        "Predicted Processing Days"
    ])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

# =========================
# SESSION STATE (SINGLE SOURCE OF TRUTH)
# =========================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = load_history()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("visa_processing_model.pkl")

model = load_model()

# =========================
# PREDICTION FUNCTION
# =========================
def predict_processing_time(
    country: str,
    visa_type: str,
    application_month: int,
    age: int,
    travel_history_count: int
) -> int:

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=model.feature_names_in_
    )

    input_df["age"] = age
    input_df["travel_history_count"] = travel_history_count
    input_df["application_month"] = application_month

    country_col = f"country_{country}"
    if country_col in input_df.columns:
        input_df[country_col] = 1

    visa_col = f"visa_type_{visa_type}"
    if visa_col in input_df.columns:
        input_df[visa_col] = 1

    prediction = model.predict(input_df)[0]
    return int(round(prediction))

# =========================
# UI
# =========================
st.title("AI-Enabled Visa Processing Time Estimator")
st.write("Estimate visa processing time using a trained ML model")

country = st.selectbox(
    "Applicant Country",
    [
        "India", "USA", "UK", "Canada", "Australia",
        "Brazil", "Germany", "France", "Italy",
        "Spain", "China", "Japan"
    ]
)

visa_type = st.selectbox(
    "Visa Type",
    ["Tourist", "Student", "Work", "Business"]
)

application_month = st.slider(
    "Application Month",
    min_value=1,
    max_value=12,
    value=1
)

age = st.number_input(
    "Applicant Age",
    min_value=0,
    max_value=100,
    value=25
)

travel_history_count = st.number_input(
    "Number of Previous Travels",
    min_value=0,
    max_value=50,
    value=2
)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict Processing Time"):
    try:
        st.session_state.last_prediction = predict_processing_time(
            country,
            visa_type,
            application_month,
            age,
            travel_history_count
        )

        st.success(
            f"Estimated Visa Processing Time: {st.session_state.last_prediction} days"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# =========================
# SAVE BUTTON (INDEPENDENT)
# =========================
if st.session_state.last_prediction is not None:
    if st.button("Save Prediction to History"):
        new_row = pd.DataFrame([{
            "Country": country,
            "Visa Type": visa_type,
            "Application Month": application_month,
            "Age": age,
            "Travel History": travel_history_count,
            "Predicted Processing Days": st.session_state.last_prediction
        }])

        st.session_state.prediction_history = pd.concat(
            [st.session_state.prediction_history, new_row],
            ignore_index=True
        )

        save_history(st.session_state.prediction_history)

        st.success("Prediction saved permanently")

# =========================
# HISTORY + CHARTS (ALWAYS VISIBLE)
# =========================
if not st.session_state.prediction_history.empty:
    st.subheader("Past Visa Processing Predictions")
    st.dataframe(st.session_state.prediction_history)

    st.subheader("Processing Time Trend")
    st.line_chart(
        st.session_state.prediction_history["Predicted Processing Days"]
    )

    csv = st.session_state.prediction_history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction History",
        data=csv,
        file_name="visa_processing_history.csv",
        mime="text/csv"
    )
