"""The churn model behind one HTTP endpoint (D03 6.1 and 6.2).

    MODEL_PATH=churn_model.joblib uvicorn app:app --port 8000
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @customer.json

Run it from this folder (D03/05-Evr-Ship-Churn-Model). The model file is the whole pipeline
(preprocessing and model) saved with joblib in step 5 of ship.ipynb, so the endpoint receives
the same 19 raw columns the training table had. THRESHOLD is the cost-matrix threshold chosen
on validation folds in 2.2.
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL = joblib.load(os.environ.get("MODEL_PATH", "churn_model.joblib"))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.278"))
app = FastAPI(title="Telco churn model")


class Customer(BaseModel):
    """One customer, in the training table's own column names and types."""
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])
    p = float(MODEL.predict_proba(X)[0, 1])
    return {"churn_probability": round(p, 3), "flag": p >= THRESHOLD, "threshold": THRESHOLD}
