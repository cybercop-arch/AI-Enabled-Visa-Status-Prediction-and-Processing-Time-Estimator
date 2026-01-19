from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Visa Processing Time Prediction API",
    description="API for predicting visa processing time using ML model"
)

# Load trained model
model = joblib.load("visa_processing_model.pkl")

class VisaRequest(BaseModel):
    country: str
    visa_type: str
    application_month: int
    age: int
    travel_history_count: int

@app.post("/predict")
def predict_processing_time(data: VisaRequest):
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=model.feature_names_in_
    )

    input_df["age"] = data.age
    input_df["travel_history_count"] = data.travel_history_count
    input_df["application_month"] = data.application_month

    country_col = f"country_{data.country}"
    if country_col in input_df.columns:
        input_df[country_col] = 1

    visa_col = f"visa_type_{data.visa_type}"
    if visa_col in input_df.columns:
        input_df[visa_col] = 1

    prediction = model.predict(input_df)[0]

    return {
        "estimated_processing_days": int(round(prediction))
    }
