"""Loads the saved model; if it is incompatible with the installed scikit-learn version it is rebuilt automatically."""
import json, os, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from hr_features import prepare

OUT = "outputs"
URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"

def _data():
    try:
        df = pd.read_csv(URL, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", encoding="utf-8-sig")
    d = prepare(df)
    return d.drop(columns=["Attrition"]), (d["Attrition"] == "Yes").astype(int)

def train_model():
    p = json.load(open(f"{OUT}/model_params.json"))
    X, y = _data()
    cat = X.select_dtypes(exclude="number").columns.tolist(); num = X.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    m = Pipeline([("pre", pre), ("clf", LogisticRegression(C=p["C"], max_iter=3000, class_weight="balanced", random_state=p["seed"]))]).fit(X, y)
    joblib.dump(m, f"{OUT}/attrition_model.joblib")
    return m

def load_model():
    meta = json.load(open(f"{OUT}/model_meta.json"))
    try:
        m = joblib.load(f"{OUT}/attrition_model.joblib")
        row = pd.DataFrame([meta["defaults"]])[meta["feature_columns"]]
        for c in meta["numeric"]: row[c] = row[c].astype(float)
        m.predict_proba(row)                      # compatibility test
        return m
    except Exception:
        return train_model()
