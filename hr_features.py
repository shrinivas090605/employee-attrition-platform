"""Shared feature engineering for the Employee Attrition platform.
Used by the notebook, the dashboard and the REST scoring client so that the
deployed model always receives identical engineered features."""
import pandas as pd

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["IncomeToLevel"] = d["MonthlyIncome"] / d["JobLevel"]
    d["TenureRatio"] = d["YearsAtCompany"] / (d["TotalWorkingYears"] + 1)
    d["PromotionGap"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["StabilityScore"] = d["YearsAtCompany"] / (d["NumCompaniesWorked"] + 1)
    d["SatisfactionIndex"] = d[["EnvironmentSatisfaction", "JobSatisfaction",
                                "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    return d

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant/ID columns and add engineered features (keeps Attrition if present)."""
    d = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return add_features(d)
