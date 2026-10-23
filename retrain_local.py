"""Scheduled retraining (run by GitHub Actions weekly). Replaces the model only if cross-validated AUC does not get worse."""
import json, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from model_utils import _data, OUT
import model_utils

X, y = _data()
meta = json.load(open(f"{OUT}/model_meta.json")); p = json.load(open(f"{OUT}/model_params.json"))
old = meta["comparison"][meta["best_model"]]["CV AUC (train)"]
model = model_utils.train_model()   # trains on the latest data and saves it
new = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=p["seed"]), scoring="roc_auc").mean()
print(f"Old CV AUC {old:.3f} | New CV AUC {new:.3f}")
if new + 0.01 < old:
    raise SystemExit("New model is worse - run failed on purpose so the model is not replaced")
meta["comparison"][meta["best_model"]]["CV AUC (train)"] = round(float(new), 3)
meta["n_employees"] = int(len(X)); json.dump(meta, open(f"{OUT}/model_meta.json", "w"), indent=2)
print("Model retrained and saved.")
