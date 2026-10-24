# Employee Attrition Analytics Platform 

| Part | Tool | File |
|---|---|---|
| Model + analysis | Jupyter notebook | Attrition_Advanced_Analytics_*.ipynb |
| Dashboard + live prediction | Streamlit Community Cloud | dashboard_app.py |
| REST API (+ auto documentation at /docs) | FastAPI on Render | api.py |
| Scheduled retraining | GitHub Actions (weekly) | .github/workflows/retrain.yml + retrain_local.py |

## 1. Upload to GitHub
Create a repo, upload ALL files in this folder (keep the `outputs` folder). Then add the workflow file in the GitHub website:
Add file > Create new file > name it `.github/workflows/retrain.yml` > paste the contents of retrain.yml from this folder > Commit.

## 2. Dashboard on Streamlit Cloud
share.streamlit.io > Create app > pick your repo > main file `dashboard_app.py` > Deploy. You get a public link.

## 3. REST API on Render
render.com > New > Web Service > connect your repo.
Build command: `pip install -r requirements.txt`   Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`   Instance: Free.
Your API: `https://<name>.onrender.com`   Documentation page: `https://<name>.onrender.com/docs`   (first call can take ~1 minute on the free plan)

### API reference
- GET  /health  -> {"status":"ok","model":"Logistic Regression","threshold":...}
- POST /predict -> body {"employees":[{"Age":24,"OverTime":"Yes","MonthlyIncome":2200,"JobLevel":1,"Department":"Sales"}]}
  response {"predictions":[{"probability_of_leaving":0.63,"risk":"High","will_leave_prediction":true}]}
  Omitted fields are filled with typical values. Errors: 422 invalid field type.

## 4. Retraining
GitHub repo > Actions tab > "Weekly model retraining" > Run workflow (shows it works). It also runs every Monday automatically.
