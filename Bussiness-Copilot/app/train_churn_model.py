"""
train_churn_model.py
Trains and compares multiple models for churn prediction:
  - Logistic Regression (simple baseline)
  - Random Forest
  - XGBoost
  - LightGBM (skipped automatically if it crashes — known Windows issue)
Logs every model's run to MLflow, then selects and saves the best one
based on F1 score (better than accuracy here since churn is imbalanced).

Run:
    python app/train_churn_model.py
Then compare visually:
    mlflow ui
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os

os.makedirs("models", exist_ok=True)

# ---- 1. Load data ----
df = pd.read_csv("data/customers.csv")
print(f"Loaded {len(df)} customers.")
print(df["churned"].value_counts())
print()

# ---- 2. Feature engineering ----
df["signup_date"] = pd.to_datetime(df["signup_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])

today = pd.Timestamp("2026-07-31")
df["days_since_signup"] = (today - df["signup_date"]).dt.days
df["days_since_last_order"] = (today - df["last_order_date"]).dt.days

le_city = LabelEncoder()
le_category = LabelEncoder()
le_coupon = LabelEncoder()

df["city_enc"] = le_city.fit_transform(df["city"])
df["category_enc"] = le_category.fit_transform(df["preferred_category"])
df["coupon_enc"] = le_coupon.fit_transform(df["used_coupon_last_order"])

df["churned_enc"] = (df["churned"] == "Yes").astype(int)

features = [
    "total_orders", "days_since_signup", "days_since_last_order",
    "support_tickets", "avg_review_rating", "total_spent_pkr",
    "city_enc", "category_enc", "coupon_enc"
]

X = df[features]
y = df["churned_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 3. Try to import LightGBM — optional, since it has a known Windows crash issue ----
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not installed — skipping it in the comparison.\n")

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"),
    "xgboost": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss"),
}
if LIGHTGBM_AVAILABLE:
    models["lightgbm"] = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)

mlflow.set_experiment("brightbyte_churn_prediction")

results = []
trained_models = {}

for name, model in models.items():
    try:
        with mlflow.start_run(run_name=name):
            if name == "lightgbm":
                # LightGBM on Windows can crash on certain pandas memory layouts —
                # converting to plain float64 numpy arrays avoids the known issue.
                model.fit(X_train.to_numpy(dtype="float64"), y_train.to_numpy(dtype="float64"))
                preds = model.predict(X_test.to_numpy(dtype="float64"))
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            if name == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            results.append({
                "model": name, "accuracy": acc, "f1_score": f1,
                "precision": precision, "recall": recall
            })
            trained_models[name] = model

            print(f"{name:22s} | acc={acc:.3f}  f1={f1:.3f}  precision={precision:.3f}  recall={recall:.3f}")

    except Exception as e:
        print(f"{name:22s} | FAILED — skipping this model. Error: {e}")
        continue

# ---- 4. Compare and pick the best model (by F1 — better for imbalanced churn data) ----
if not results:
    raise RuntimeError("All models failed to train. Check the errors above.")

results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
print("\n=== Model Comparison (sorted by F1 score) ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]
print(f"\nBest model: {best_model_name}")

best_model = trained_models[best_model_name]
if best_model_name == "lightgbm":
    test_input = X_test.to_numpy(dtype="float64")
else:
    test_input = X_test

print("\nClassification report for best model:")
print(classification_report(y_test, best_model.predict(test_input), target_names=["Retained", "Churned"]))

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    print("Feature importances:")
    print(importances)

# ---- 5. Save the best model + encoders ----
joblib.dump(best_model, "models/churn_model.pkl")
joblib.dump(le_city, "models/le_city.pkl")
joblib.dump(le_category, "models/le_category.pkl")
joblib.dump(le_coupon, "models/le_coupon.pkl")

with open("models/best_model_name.txt", "w") as f:
    f.write(best_model_name)

print(f"\nSaved best model ({best_model_name}) to models/churn_model.pkl")
print("Run 'mlflow ui' in a separate terminal to visually compare all runs at http://127.0.0.1:5000")