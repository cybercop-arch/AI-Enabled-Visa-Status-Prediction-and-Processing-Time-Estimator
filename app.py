import streamlit as st
import pandas as pd
import joblib
import os

HISTORY_FILE = "prediction_history.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = load_history()

# Page configuration
st.set_page_config(
    page_title="Visa Processing Time Estimator",
    layout="centered"
)

st.title("AI-Enabled Visa Processing Time Estimator")
st.write("Estimate visa processing time using a trained ML model")

# Load trained model 
@st.cache_resource
def load_model():
    return joblib.load("visa_processing_model.pkl")

model = load_model()

# Prediction function
def predict_processing_time(
    country: str,
    visa_type: str,
    application_month: int,
    age: int,
    travel_history_count: int
) -> int:
    """
    Creates an input dataframe that exactly matches
    the training feature schema and runs prediction.
    """

    # Create empty input with correct feature order
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=model.feature_names_in_
    )

    # Numerical features
    input_df["age"] = age
    input_df["travel_history_count"] = travel_history_count
    input_df["application_month"] = application_month

    # One-hot country
    country_col = f"country_{country}"
    if country_col in input_df.columns:
        input_df[country_col] = 1

    # One-hot visa type
    visa_col = f"visa_type_{visa_type}"
    if visa_col in input_df.columns:
        input_df[visa_col] = 1

    prediction = model.predict(input_df)[0]
    return int(round(prediction))

# UI INPUTS
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

# SESSION STATE INITIALIZATION
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# PREDICT BUTTON
if st.button("Predict Processing Time"):
    try:
        predicted_days = predict_processing_time(
            country,
            visa_type,
            application_month,
            age,
            travel_history_count
        )

        st.session_state.last_prediction = predicted_days

        st.success(
            f"Estimated Visa Processing Time: {predicted_days} days"
        )

    except Exception as e:
        st.error("Prediction failed. Check model compatibility.")
        st.exception(e)

# SAVE BUTTON (independent)
if st.session_state.last_prediction is not None:
    if st.button("Save Prediction to History"):
        new_row = {
            "Country": country,
            "Visa Type": visa_type,
            "Application Month": application_month,
            "Age": age,
            "Travel History": travel_history_count,
            "Predicted Processing Days": st.session_state.last_prediction
        }

        st.session_state.prediction_history = pd.concat(
            [
                st.session_state.prediction_history,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        save_history(st.session_state.prediction_history)

        st.success("Prediction saved permanently")


# ALWAYS SHOW HISTORY & CHARTS
if st.session_state.prediction_history:
    st.subheader("Past Visa Processing Predictions")

    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df)

    st.subheader("Processing Time Trend")
    st.line_chart(history_df["Predicted Processing Days"])

    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prediction History",
        data=csv,
        file_name="visa_processing_history.csv",
        mime="text/csv"
    )

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