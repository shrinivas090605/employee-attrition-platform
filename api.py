"""REST API for the attrition model.  Run: uvicorn api:app --reload   Docs (Swagger UI): /docs"""
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from hr_features import prepare
from model_utils import load_model

meta = json.load(open("outputs/model_meta.json"))
model = load_model()
app = FastAPI(title="Employee Attrition Prediction API", version="1.0",
              description="Predicts the probability that an employee will leave. Send raw HR fields; any field you omit is filled with a typical value.")

class Employee(BaseModel):
    Age: int | None = Field(None, example=29)
    MonthlyIncome: int | None = Field(None, example=3200)
    OverTime: str | None = Field(None, example="Yes")
    JobLevel: int | None = Field(None, example=1)
    JobSatisfaction: int | None = Field(None, example=2)
    WorkLifeBalance: int | None = Field(None, example=2)
    YearsAtCompany: int | None = Field(None, example=2)
    YearsSinceLastPromotion: int | None = Field(None, example=2)
    Department: str | None = Field(None, example="Sales")
    BusinessTravel: str | None = Field(None, example="Travel_Frequently")
    model_config = {"extra": "allow"}

class Batch(BaseModel):
    employees: list[Employee]

def _score(records):
    rows = []
    for r in records:
        d = dict(meta["defaults"]); d.update({k: v for k, v in r.items() if v is not None}); rows.append(d)
    X = pd.DataFrame(rows)
    for c in X.columns:
        if c in meta["numeric"]: X[c] = X[c].astype(float)
    # defaults hold engineered features; recompute them from the raw fields
    raw = X.drop(columns=[c for c in ["IncomeToLevel", "TenureRatio", "PromotionGap", "StabilityScore", "SatisfactionIndex"] if c in X.columns])
    X = prepare(raw.assign(Attrition="No")).drop(columns=["Attrition"])[meta["feature_columns"]]
    p = model.predict_proba(X)[:, 1]
    band = lambda v: "High" if v >= meta["threshold"] else "Medium" if v >= meta["risk_bands"]["low_max"] else "Low"
    return [{"probability_of_leaving": round(float(v), 4), "risk": band(v), "will_leave_prediction": bool(v >= meta["threshold"])} for v in p]

@app.get("/health")
def health():
    return {"status": "ok", "model": meta["best_model"], "threshold": meta["threshold"]}

@app.post("/predict")
def predict(batch: Batch):
    return {"predictions": _score([e.model_dump() for e in batch.employees])}
