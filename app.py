"""
Production FastAPI microservice for Car Insurance Claim Prediction.

Provides:
  - GET  /         --> Interactive Web Dashboard (HTML5/CSS3)
  - POST /predict  --> Machine Learning Inference Endpoint (REST JSON)
  - GET  /health   --> Liveness / Readiness Container Health Probe
  - GET  /docs     --> Swagger / OpenAPI Documentation

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pipeline.joblib")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load serialized preprocessing and model pipeline on service startup."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model pipeline not found at {MODEL_PATH}. "
            "Run 'python car_insurance_analysis.py' first to train and serialize the pipeline."
        )
    pipeline = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Auto Insurance Claim Risk Prediction API",
    description="Actuarial risk scoring microservice for automobile insurance policies.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external frontends or multi-service deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
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
    )

    age: int = Field(..., ge=0, le=3, description="Age group: 0 (16-25), 1 (26-39), 2 (40-64), 3 (65+)")
    gender: int = Field(..., ge=0, le=1, description="Gender: 0 (female), 1 (male)")
    driving_experience: str = Field(..., description="Driving experience bracket: '0-9y', '10-19y', '20-29y', '30y+'")
    education: str = Field(..., description="Education: 'none', 'high school', 'university'")
    income: str = Field(..., description="Income group: 'poverty', 'working class', 'middle class', 'upper class'")
    credit_score: float | None = Field(None, ge=0.0, le=1.0, description="Credit score between 0 and 1 (optional)")
    vehicle_ownership: float = Field(..., ge=0.0, le=1.0, description="Vehicle ownership: 1.0 (owns car), 0.0 (does not)")
    vehicle_year: str = Field(..., description="Vehicle year: 'before 2015' or 'after 2015'")
    married: float = Field(..., ge=0.0, le=1.0, description="Marital status: 1.0 (married), 0.0 (single)")
    children: float = Field(..., ge=0.0, le=1.0, description="Has children: 1.0 (yes), 0.0 (no)")
    postal_code: int = Field(..., description="Postal code integer: 10238, 21217, 32765, 92101")
    annual_mileage: float | None = Field(None, ge=0.0, description="Annual mileage in miles (optional)")
    vehicle_type: str = Field(..., description="Vehicle type: 'sedan' or 'sports car'")
    speeding_violations: int = Field(..., ge=0, description="Number of speeding violations")
    duis: int = Field(..., ge=0, description="Number of DUIs")
    past_accidents: int = Field(..., ge=0, description="Number of past accidents")


class PredictionResponse(BaseModel):
    claim_probability: float
    predicted_claim: int
    risk_tier: str
    missing_fields_imputed: list[str]


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the embedded interactive web UI."""
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Web dashboard template not found. Visit <a href='/docs'>/docs</a> for API.</h2>", status_code=200)


@app.get("/health")
def health_check():
    """Liveness probe for Docker container health checks and Kubernetes/cloud orchestrators."""
    return {
        "status": "healthy",
        "service": "auto-insurance-claim-risk-prediction",
        "model_loaded": pipeline is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_claim(payload: CustomerFeatures):
    """
    Accepts customer profile JSON, applies pipeline preprocessing,
    and returns predicted claim probability and risk classification.
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

    missing_imputed = []
    if payload.credit_score is None:
        missing_imputed.append("credit_score")
    if payload.annual_mileage is None:
        missing_imputed.append("annual_mileage")

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
    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference error during pipeline execution: {e!s}",
        ) from e
