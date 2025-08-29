# Customer Churn Prediction (WGU Task 2)

This repo contains a fully functional ML workflow that predicts customer churn using the Kaggle **Telco Customer Churn** dataset.

## Dataset Citation
IBM Sample Data — *Telco Customer Churn*. Available on Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn

## Requirements (C1)
- OS: Windows 11 (local) and/or Linux (cloud)
- Hardware: 16 GB RAM recommended; GPU optional
- Software:
  - Python 3.10+
  - `pip install -r requirements.txt`
  - Git

## Project Structure
```
data/raw/           # original CSV
data/processed/     # preprocessed CSV + pipeline
src/preprocess.py   # preprocessing (B1)
src/train.py        # train & evaluate (B2-B4)
src/optimize.py     # CV + hyperparam tuning (B5-B6)
models/             # saved models
```

## How to Run (C2)
1. Place the raw CSV in `data/raw/` as `Telco-Customer-Churn.csv`.
2. (Optional) Create a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Preprocess data (saves processed CSV + pipeline):
   ```bash
   python src/preprocess.py
   ```
5. Train baseline model and print metrics:
   ```bash
   python src/train.py
   ```
6. Cross-validate and hyperparameter tune XGBoost:
   ```bash
   python src/optimize.py
   ```

## Artifacts
- `data/processed/telco_churn_processed.csv`
- `data/processed/preprocess_pipeline.joblib`
- `models/baseline_logreg.joblib`
- `models/best_model.joblib`

## Notes
- Metrics reported: **accuracy, precision, recall, F1**.
- Class imbalance handled via `class_weight='balanced'` and CV F1 scoring.