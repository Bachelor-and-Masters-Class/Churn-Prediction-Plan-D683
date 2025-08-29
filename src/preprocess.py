import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

csv_path = RAW / "Telco-Customer-Churn.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Place the raw dataset at {csv_path}")

df = pd.read_csv(csv_path)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

target_col = "Churn"
if target_col not in df.columns:
    raise KeyError(f"Expected target column '{target_col}' not found.")

y = (df[target_col].astype(str).str.lower().eq("yes")).astype(int)
drop_cols = [c for c in ["customerID", "customer_Id", "CustomerID"] if c in df.columns]
X = df.drop(columns=[target_col] + drop_cols)

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])

cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])

X_processed = preprocess.fit_transform(X)

Xp = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
proc_df = pd.DataFrame(Xp)
proc_df["Churn"] = y.values
proc_df.to_csv(PROCESSED / "telco_churn_processed.csv", index=False)

joblib.dump(preprocess, PROCESSED / "preprocess_pipeline.joblib")
print("Saved processed CSV and pipeline.")