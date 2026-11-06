# Customer Churn Prediction Plan

**WGU D683 · Advanced AI and ML · Task 1**

The project plan for a model that predicts which subscription customers are likely to cancel, so the business can step in before they leave.

## The plan

- **Problem:** high customer churn raises costs, and without a prediction tool at-risk customers aren't spotted in time.
- **Goal:** reach at least 80% AUC-ROC on test data, helping cut churn by 10% within 6 months.
- **Data:** the Kaggle Telco Customer Churn dataset: 7,043 customers with tenure, contract type, monthly charges, payment method, services and a churn label.
- **Models:** logistic regression, random forest and XGBoost, with feature importance and SHAP values to explain predictions.
- **Scoring:** AUC-ROC, precision, recall and lift.
- **Requirements:** take customer data in and give a churn risk score out, in under 2 seconds per customer, for at least 50,000 records.
- **Risks:** class imbalance (handled with SMOTE and class weights), overfitting (cross-validation and regularization) and data quality (cleaning and input checks).
- **Schedule:** 5 phases from Sept 1 to Oct 15, 2025: scoping, data prep, training, evaluation and final report.

## Files

- `1793936712105_Advanced_AIML_Topic_Approval_Form.pdf`: the signed topic approval form

**Built with:** Python, scikit-learn, XGBoost, pandas, TensorFlow, Jupyter, Google Colab
