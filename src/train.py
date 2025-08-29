import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

PROCESSED = Path("data/processed")
csv_path = PROCESSED / "telco_churn_processed.csv"
if not csv_path.exists():
    raise FileNotFoundError("Run preprocess first.")

df = pd.read_csv(csv_path)
target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
}
print(metrics)
print("\nClassification report:\n", classification_report(y_test, y_pred))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/baseline_logreg.joblib")
print("Saved model to models/baseline_logreg.joblib")