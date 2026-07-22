"""
Production FastAPI microservice for Car Insurance Claim Prediction.

Single-endpoint deployment accepting customer demographic and vehicle data,
applying the fitted preprocessing pipeline, and returning claim probability
and predicted outcome class.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==============================================================================
# INFERENCE-TIME EDGE CASE HANDLING DOCUMENTATION
# ==============================================================================
# 1. Missing Values (credit_score and annual_mileage):
#    - The scikit-learn pipeline incorporates a SimpleImputer(strategy='median')
#      fitted strictly on the training set.
#    - If credit_score or annual_mileage is null/None/NaN in the incoming JSON,
#      it is automatically replaced with the training median (~0.525 for credit_score,
#      ~12,000 for annual_mileage) before scaling and inference.
#
# 2. Unseen Categorical Levels:
#    - Nominal features (vehicle_year, vehicle_type, postal_code) are handled by
#      OneHotEncoder(handle_unknown='ignore', drop='first'). If an unknown postal
#      code or vehicle type arrives, all one-hot indicator columns for that feature
#      default to 0 (baseline category behavior) without raising an exception.
#    - Ordinal features (driving_experience, education, income) are handled by
#      OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1). Any
#      unseen category is mapped to -1 without causing runtime failure.
# ==============================================================================

app = FastAPI(
    title="Car Insurance Claim Prediction API",
    description="Minimal production endpoint predicting auto insurance claim risk.",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
pipeline = None


@app.on_event("startup")
def load_model():
    """Load serialized preprocessing and model pipeline on service startup."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model pipeline not found at {MODEL_PATH}. "
            "Run 'python car_insurance_analysis.py' first to train and serialize the pipeline."
        )
    pipeline = joblib.load(MODEL_PATH)


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=0, le=3, description="Age group: 0 (16-25), 1 (26-39), 2 (40-64), 3 (65+)")
    gender: int = Field(..., ge=0, le=1, description="Gender: 0 (female), 1 (male)")
    driving_experience: str = Field(..., description="Driving experience bracket: '0-9y', '10-19y', '20-29y', '30y+'")
    education: str = Field(..., description="Education: 'none', 'high school', 'university'")
    income: str = Field(..., description="Income group: 'poverty', 'working class', 'middle class', 'upper class'")
    credit_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Credit score between 0 and 1 (optional)")
    vehicle_ownership: float = Field(..., ge=0.0, le=1.0, description="Vehicle ownership: 1.0 (owns car), 0.0 (does not)")
    vehicle_year: str = Field(..., description="Vehicle year: 'before 2015' or 'after 2015'")
    married: float = Field(..., ge=0.0, le=1.0, description="Marital status: 1.0 (married), 0.0 (single)")
    children: float = Field(..., ge=0.0, le=1.0, description="Has children: 1.0 (yes), 0.0 (no)")
    postal_code: int = Field(..., description="Postal code integer: 10238, 21217, 32765, 92101")
    annual_mileage: Optional[float] = Field(None, ge=0.0, description="Annual mileage in miles (optional)")
    vehicle_type: str = Field(..., description="Vehicle type: 'sedan' or 'sports car'")
    speeding_violations: int = Field(..., ge=0, description="Number of speeding violations")
    duis: int = Field(..., ge=0, description="Number of DUIs")
    past_accidents: int = Field(..., ge=0, description="Number of past accidents")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 2,
                "gender": 1,
                "driving_experience": "20-29y",
                "education": "university",
                "income": "upper class",
                "credit_score": 0.629,
                "vehicle_ownership": 1.0,
                "vehicle_year": "after 2015",
                "married": 1.0,
                "children": 1.0,
                "postal_code": 10238,
                "annual_mileage": 12000.0,
                "vehicle_type": "sedan",
                "speeding_violations": 0,
                "duis": 0,
                "past_accidents": 0,
            }
        }


class PredictionResponse(BaseModel):
    claim_probability: float
    predicted_claim: int
    risk_tier: str
    missing_fields_imputed: list[str]


@app.get("/")
def health_check():
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "service": "car-insurance-claim-predictor",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_claim(payload: CustomerFeatures):
    """
    Accepts customer profile JSON, applies pipeline preprocessing,
    and returns predicted claim probability and classification.
    """
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=500,
                detail="Model pipeline has not been trained or saved yet.",
            )

    # Track missing fields for transparency
    missing_imputed = []
    if payload.credit_score is None:
        missing_imputed.append("credit_score")
    if payload.annual_mileage is None:
        missing_imputed.append("annual_mileage")

    # Format into DataFrame with correct columns for ColumnTransformer
    input_data = {
        "age": [payload.age],
        "gender": [payload.gender],
        "driving_experience": [payload.driving_experience],
        "education": [payload.education],
        "income": [payload.income],
        "credit_score": [np.nan if payload.credit_score is None else payload.credit_score],
        "vehicle_ownership": [payload.vehicle_ownership],
        "vehicle_year": [payload.vehicle_year],
        "married": [payload.married],
        "children": [payload.children],
        "postal_code": [payload.postal_code],
        "annual_mileage": [np.nan if payload.annual_mileage is None else payload.annual_mileage],
        "vehicle_type": [payload.vehicle_type],
        "speeding_violations": [payload.speeding_violations],
        "duis": [payload.duis],
        "past_accidents": [payload.past_accidents],
    }
    input_df = pd.DataFrame(input_data)

    try:
        prob = float(pipeline.predict_proba(input_df)[0, 1])
        pred = int(pipeline.predict(input_df)[0])

        if prob >= 0.70:
            risk_tier = "High"
        elif prob >= 0.40:
            risk_tier = "Moderate"
        else:
            risk_tier = "Low"

        return PredictionResponse(
            claim_probability=round(prob, 4),
            predicted_claim=pred,
            risk_tier=risk_tier,
            missing_fields_imputed=missing_imputed,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference error during pipeline execution: {str(e)}",
        )
