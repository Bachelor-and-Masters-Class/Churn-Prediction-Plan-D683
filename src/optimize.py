import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
import joblib

PROCESSED = Path("data/processed")
csv_path = PROCESSED / "telco_churn_processed.csv"
if not csv_path.exists():
    raise FileNotFoundError("Run preprocess first.")

df = pd.read_csv(csv_path)
target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)
scores = cross_val_score(xgb, X, y, scoring="f1", cv=cv, n_jobs=1)
print("5-fold F1 scores:", scores)
print("Mean F1:", scores.mean())

param_grid = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1]
}
grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=make_scorer(f1_score),
    cv=cv,
    n_jobs=1,
    verbose=2
)
grid.fit(X, y)

print("Best params:", grid.best_params_)
print("Best F1:", grid.best_score_)

Path("models").mkdir(exist_ok=True)
joblib.dump(grid.best_estimator_, "models/best_model.joblib")
print("Saved tuned model to models/best_model.joblib")