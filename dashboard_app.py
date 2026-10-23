"""HR Attrition Analytics Dashboard (Streamlit).
Run:  streamlit run dashboard_app.py
Needs the files produced by the notebook in ./outputs (dashboard_data.csv, model_meta.json, attrition_model.joblib).
The prediction tab scores with the saved model."""
import os, json
import joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from hr_features import prepare

OUT = "outputs"
st.set_page_config(page_title="HR Attrition Analytics", layout="wide")

@st.cache_data
def load():
    return pd.read_csv(f"{OUT}/dashboard_data.csv"), json.load(open(f"{OUT}/model_meta.json"))
df, meta = load()
from model_utils import load_model
model = st.cache_resource(load_model)()

st.title("Employee Attrition Analytics Platform")
st.sidebar.header("Filters")
dept = st.sidebar.multiselect("Department", sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
risk = st.sidebar.multiselect("Risk category", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
f = df[df["Department"].isin(dept) & df["RiskCategory"].isin(risk)]

tabs = st.tabs(["HR KPIs", "Attrition & Risk", "Workforce Demographics", "Model Performance", "Predict Attrition"])

with tabs[0]:
    n = len(f); left = int(f["AttritionFlag"].sum())
    c = st.columns(5)
    c[0].metric("Headcount", n); c[1].metric("Employees left", left)
    c[2].metric("Attrition rate", f"{100*left/max(n,1):.1f}%"); c[3].metric("Retention rate", f"{100*(1-left/max(n,1)):.1f}%")
    c[4].metric("High-risk employees", int((f["RiskCategory"] == "High").sum()))
    g = f.groupby("Department").agg(Headcount=("AttritionFlag", "size"), Left=("AttritionFlag", "sum"), AvgRisk=("RiskScore", "mean")).reset_index()
    g["Attrition rate %"] = (100 * g["Left"] / g["Headcount"]).round(1)
    st.subheader("Department-wise KPIs"); st.dataframe(g.round(3), use_container_width=True, hide_index=True)
    t = f.groupby("YearsAtCompany")["AttritionFlag"].mean().mul(100).reset_index(name="Attrition rate %")
    st.plotly_chart(px.line(t, x="YearsAtCompany", y="Attrition rate %", markers=True, title="Attrition rate by tenure (years at company)"), use_container_width=True)

with tabs[1]:
    a, b = st.columns(2)
    d = f.groupby("Department")["AttritionFlag"].mean().mul(100).reset_index(name="Attrition rate %")
    a.plotly_chart(px.bar(d, x="Department", y="Attrition rate %", title="Attrition by department", text_auto=".1f"), use_container_width=True)
    rc = f["RiskCategory"].value_counts().reindex(["Low", "Medium", "High"]).reset_index(); rc.columns = ["Risk", "Employees"]
    b.plotly_chart(px.pie(rc, names="Risk", values="Employees", hole=.5, title="Employee risk categories", color="Risk",
                          color_discrete_map={"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}), use_container_width=True)
    jr = f.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values().reset_index(name="Attrition rate %")
    st.plotly_chart(px.bar(jr, x="Attrition rate %", y="JobRole", orientation="h", title="Attrition by job role"), use_container_width=True)
    st.subheader("Highest-risk employees (out-of-fold scores)")
    cols = [c for c in ["EmployeeNumber", "Department", "JobRole", "Age", "OverTime", "MonthlyIncome", "YearsAtCompany", "RiskScore", "RiskCategory"] if c in f.columns]
    st.dataframe(f.sort_values("RiskScore", ascending=False)[cols].head(25).round(3), use_container_width=True, hide_index=True)

with tabs[2]:
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(f, x="Age", color="Attrition", nbins=25, barmode="overlay", title="Age distribution"), use_container_width=True)
    b.plotly_chart(px.pie(f, names="Gender", title="Gender split", hole=.4), use_container_width=True)
    a.plotly_chart(px.histogram(f, x="MaritalStatus", color="Attrition", barmode="group", title="Marital status"), use_container_width=True)
    b.plotly_chart(px.histogram(f, x="YearsAtCompany", nbins=20, title="Tenure distribution"), use_container_width=True)
    a.plotly_chart(px.histogram(f, x="OverTime", color="Attrition", barmode="group", title="Overtime vs attrition"), use_container_width=True)
    b.plotly_chart(px.box(f, x="Attrition", y="MonthlyIncome", title="Monthly income vs attrition"), use_container_width=True)

with tabs[3]:
    st.subheader(f"Selected model: {meta['best_model']}  |  decision threshold {meta['threshold']:.2f}")
    m = pd.DataFrame({"Default (0.50)": meta["test_metrics_default"], f"Tuned ({meta['threshold']:.2f})": meta["test_metrics_tuned"]}).T.round(3)
    st.dataframe(m, use_container_width=True)
    comp = pd.DataFrame(meta["comparison"]).T.reset_index().rename(columns={"index": "Model"})
    st.plotly_chart(px.bar(comp, x="ROC-AUC", y="Model", orientation="h", title="Test ROC-AUC by model (NN vs classical vs Bayes vs Discriminant)"), use_container_width=True)
    st.dataframe(comp.round(3), use_container_width=True, hide_index=True)
    for img, cap in [("evaluation.png", "Confusion matrix and precision/recall"), ("model_comparison.png", "ROC curves")]:
        if os.path.exists(f"{OUT}/{img}"): st.image(f"{OUT}/{img}", caption=cap)

with tabs[4]:
    st.subheader("Predict attrition risk for an employee")
    d0 = meta["defaults"]; c = st.columns(3)
    age = c[0].slider("Age", 18, 60, int(float(d0["Age"]))); income = c[1].number_input("Monthly income", 1000, 25000, int(float(d0["MonthlyIncome"])))
    ot = c[2].selectbox("OverTime", ["No", "Yes"]); level = c[0].slider("Job level", 1, 5, int(float(d0["JobLevel"])))
    sat = c[1].slider("Job satisfaction (1-4)", 1, 4, int(float(d0["JobSatisfaction"]))); wlb = c[2].slider("Work-life balance (1-4)", 1, 4, int(float(d0["WorkLifeBalance"])))
    yrs = c[0].slider("Years at company", 0, 40, int(float(d0["YearsAtCompany"]))); promo = c[1].slider("Years since last promotion", 0, 15, int(float(d0["YearsSinceLastPromotion"])))
    dep = c[2].selectbox("Department", sorted(df["Department"].unique())); travel = c[0].selectbox("Business travel", sorted(df["BusinessTravel"].unique()))
    row = {k: (float(v) if k in meta["numeric"] else v) for k, v in d0.items()}
    raw = {k: v for k, v in row.items() if k in df.columns}
    raw.update(Age=age, MonthlyIncome=income, OverTime=ot, JobLevel=level, JobSatisfaction=sat, WorkLifeBalance=wlb,
               YearsAtCompany=yrs, YearsSinceLastPromotion=promo, Department=dep, BusinessTravel=travel)
    if st.button("Predict"):
        rec = pd.DataFrame([raw]); X = prepare(rec.assign(Attrition="No")).drop(columns=["Attrition"])[meta["feature_columns"]]
        p = float(model.predict_proba(X)[0, 1]); src = "deployed model"
        band = "High" if p >= meta["threshold"] else "Medium" if p >= meta["risk_bands"]["low_max"] else "Low"
        st.metric("Probability of leaving", f"{p:.1%}"); st.write(f"**Risk category: {band}**  (scored via {src})")
