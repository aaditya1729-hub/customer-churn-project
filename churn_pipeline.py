# churn_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt


# 1. Load Data

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Initial shape:", df.shape)


# 2. Data Cleaning

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)


# 3. Feature Engineering

df["tenure_volatility"] = df["MonthlyCharges"].std() / (df["tenure"] + 1)
df["contract_transition"] = np.where(df["Contract"] == "Month-to-month", 1, 0)
np.random.seed(42)
df["support_ticket_freq"] = np.random.randint(0, 5, df.shape[0])


# 4. Encode Categorical Variables

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    if col != "Churn":  # Leave target variable untouched
        df[col] = le.fit_transform(df[col])


# 5. Train-Test Split

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Model Training (XGBoost)

xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("XGBoost AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 7. Feature Importance - SHAP

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Save SHAP summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
print("SHAP summary plot saved as shap_summary.png")

# Save SHAP feature importance as CSV
shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)
shap_df.to_csv("shap_feature_importance.csv", index=False)
print("SHAP feature importance exported to shap_feature_importance.csv")


# 8. Export Predictions for Power BI

customer_ids = df.loc[X_test.index, "customerID"].values

df_results = X_test.copy()
df_results.insert(0, "customerID", customer_ids)  
df_results["Actual"] = y_test.values
df_results["Predicted"] = y_pred
df_results["Predicted_Prob"] = xgb_model.predict_proba(X_test)[:, 1]
df_results.to_csv("churn_predictions.csv", index=False)
print("Predictions exported to churn_predictions.csv with customerID")


# 9. Export merged dataset for Power BI (no relationship needed)

df_merged = df.loc[X_test.index].copy()
df_merged["Predicted"] = y_pred
df_merged["Predicted_Prob"] = xgb_model.predict_proba(X_test)[:, 1]
df_merged.to_csv("churn_data_with_predictions.csv", index=False)
print("Merged dataset exported to churn_data_with_predictions.csv")
