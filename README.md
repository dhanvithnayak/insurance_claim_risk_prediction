# Auto Insurance Claim Risk Prediction & Propensity Engine

An end-to-end, actuarially sound machine learning pipeline and inference microservice that predicts the probability of personal auto insurance policyholders filing a claim. Evaluated via Stratified 5-Fold Cross-Validation, explained with SHAP interpretability, and deployed via a low-latency FastAPI REST API.

---

## Key Performance Highlights

- **PR-AUC**: **`0.8361 ± 0.0112`** (+166.9% relative lift over the **`0.3133`** random-guessing baseline).
- **Claim Recall**: **`86.69% ± 0.0112`** (captures nearly 9 out of 10 claimants, minimizing unpriced risk).
- **Cross-Validated Accuracy**: **`84.01% ± 0.0048`** (+15.34% over the 68.67% naive majority-class baseline).
- **Dominant Risk Factor**: `driving_experience` is the #1 protective factor (Mean |SHAP| = 1.66, coef = -1.96).

---

## Cross-Validation Model Benchmark (5-Fold Stratified)

| Model | Accuracy (vs 68.67%) | Precision (Claim=1) | Recall (Claim=1) | F1-Score (Claim=1) | PR-AUC (vs 0.3133) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Balanced) [WINNER]** | **0.8401 ± 0.0048** | 0.6969 ± 0.0082 | **0.8669 ± 0.0112** | **0.7726 ± 0.0061** | **0.8361 ± 0.0112** |
| Random Forest (Balanced) | 0.8328 ± 0.0064 | **0.7110 ± 0.0106** | 0.7858 ± 0.0091 | 0.7465 ± 0.0090 | 0.7901 ± 0.0101 |
| XGBoost (Weighted) | 0.8330 ± 0.0049 | 0.7036 ± 0.0081 | 0.8072 ± 0.0131 | 0.7518 ± 0.0074 | 0.8087 ± 0.0070 |

---

## Statistical Data Quality & Engineering

1. **Missingness Mechanism ($\chi^2$ Test)**:
   - `credit_score` (9.8% missing, $p = 0.8189$) and `annual_mileage` (9.6% missing, $p = 0.0966$) are unassociated with claims (MCAR/MAR). Imputed using medians inside the cross-validation folds to prevent data leakage.
2. **Ordinal Encoding Justification**:
   - `income` (poverty 65.4% $\to$ working class 45.3% $\to$ middle class 27.7% $\to$ upper class 13.4%) and `education` (none 47.2% $\to$ high school 32.3% $\to$ university 22.6%) exhibit strict monotonic claim risk decay, encoded ordinally with a single degree of freedom.
3. **Postal Code Territory Audit**:
   - 4 discrete ZIP codes. Correctly one-hot encoded to avoid false continuous geometry; flagged ZIP `21217` which has a 100% claim rate ($N = 120$).

---

## Interpretability (SHAP Values)

1. `driving_experience` (Mean |SHAP| = 1.66, coef = -1.96)
2. `vehicle_ownership` (Mean |SHAP| = 0.83, coef = -1.85)
3. `vehicle_year_before 2015` (Mean |SHAP| = 0.75, coef = +1.90)
4. `gender` (Mean |SHAP| = 0.52, coef = +1.04)
5. `postal_code_32765` (Mean |SHAP| = 0.44, coef = +1.27)
6. `annual_mileage` (Mean |SHAP| = 0.26, coef = +0.34)
7. `postal_code_21217` (Mean |SHAP| = 0.24, coef = +7.68)
8. `married` (Mean |SHAP| = 0.19, coef = -0.38)

---

## Project Structure

```text
├── car_insurance_analysis.py   # Complete modeling & evaluation pipeline (Tasks 1-6)
├── app.py                      # FastAPI inference microservice
├── requirements.txt            # Dependency requirements
├── model_pipeline.joblib       # Serialized production pipeline
├── car_insurance.csv           # 10,000 policyholder records
├── data/
│   └── car_insurance.csv       # Mirrored raw dataset
└── .gitignore                  # Git exclusions
```

---

## Quickstart

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Analysis Pipeline
```bash
python car_insurance_analysis.py
```

### 3. Launch REST API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Test Prediction Endpoint
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "past_accidents": 0
  }'
```
Response:
```json
{
  "claim_probability": 0.0069,
  "predicted_claim": 0,
  "risk_tier": "Low",
  "missing_fields_imputed": []
}
```
