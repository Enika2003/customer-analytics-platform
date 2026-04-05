import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
DB_URL = "postgresql+psycopg2://localhost/retail_analytics"
engine = create_engine(DB_URL)

# -------------------------------------------------------------
# 1. Temporal Split — avoids data leakage
#    Features built on: everything BEFORE cutoff
#    Churn label:       no purchase AFTER cutoff
# -------------------------------------------------------------
print("Loading transactions...")
query = "SELECT customer_id, date_id, invoice_id, revenue, stockcode FROM fact_sales"
df = pd.read_sql(query, engine)
df["date_id"] = pd.to_datetime(df["date_id"])

max_date = df["date_id"].max()
cutoff   = max_date - pd.Timedelta(days=90)   # last 90 days = observation window
print(f"  Dataset range : {df['date_id'].min().date()} → {max_date.date()}")
print(f"  Cutoff date   : {cutoff.date()}")

# Split
hist   = df[df["date_id"] <  cutoff]   # feature period
future = df[df["date_id"] >= cutoff]   # label period

# -------------------------------------------------------------
# 2. Build Features from HISTORICAL period only
# -------------------------------------------------------------
print("\nBuilding features from historical period...")
features = hist.groupby("customer_id").agg(
    frequency       = ("invoice_id",  "nunique"),
    monetary        = ("revenue",     "sum"),
    avg_order_value = ("revenue",     "mean"),
    unique_products = ("stockcode",   "nunique"),
    last_purchase   = ("date_id",     "max"),
    first_purchase  = ("date_id",     "min"),
).reset_index()

features["recency_to_cutoff"] = (cutoff - features["last_purchase"]).dt.days
features["tenure_days"]       = (features["last_purchase"] - features["first_purchase"]).dt.days
features["avg_days_between"]  = (features["tenure_days"] / features["frequency"]).replace([np.inf], 0)

# -------------------------------------------------------------
# 3. Build Churn Label from FUTURE period
# -------------------------------------------------------------
active_in_future = future["customer_id"].unique()
features["churned"] = (~features["customer_id"].isin(active_in_future)).astype(int)

print(f"  Customers in historical period : {len(features):,}")
print(f"  Churned (no future purchase)   : {features['churned'].sum():,}  ({features['churned'].mean()*100:.1f}%)")
print(f"  Retained                       : {(1-features['churned']).sum():,}  ({(1-features['churned']).mean()*100:.1f}%)")

# -------------------------------------------------------------
# 4. Train / Test Split
# -------------------------------------------------------------
FEATURES = ["recency_to_cutoff", "frequency", "monetary", "avg_order_value",
            "unique_products", "tenure_days", "avg_days_between"]

X = features[FEATURES].fillna(0)
y = features["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# -------------------------------------------------------------
# 5. Logistic Regression
# -------------------------------------------------------------
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_preds = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"  AUC: {lr_auc:.3f}")
print(classification_report(y_test, lr_preds, target_names=["Retained", "Churned"]))

# -------------------------------------------------------------
# 6. XGBoost
# -------------------------------------------------------------
print("Training XGBoost...")
xgb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"  AUC: {xgb_auc:.3f}")
print(classification_report(y_test, xgb_preds, target_names=["Retained", "Churned"]))

# -------------------------------------------------------------
# 7. Plots
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

RocCurveDisplay.from_predictions(y_test, lr_proba,  name=f"Logistic (AUC={lr_auc:.2f})",  ax=axes[0])
RocCurveDisplay.from_predictions(y_test, xgb_proba, name=f"XGBoost  (AUC={xgb_auc:.2f})", ax=axes[0])
axes[0].set_title("ROC Curves — Churn Model")

best_preds = xgb_preds if xgb_auc >= lr_auc else lr_preds
ConfusionMatrixDisplay.from_predictions(
    y_test, best_preds,
    display_labels=["Retained", "Churned"],
    ax=axes[1], colorbar=False
)
axes[1].set_title("Confusion Matrix (Best Model)")

importances = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values()
importances.plot(kind="barh", ax=axes[2], color="#2196F3")
axes[2].set_title("Feature Importance (XGBoost)")

plt.tight_layout()
plt.savefig("data/churn_model.png", dpi=150)
print("Saved: data/churn_model.png")

# -------------------------------------------------------------
# 8. Score ALL historical customers & write to PostgreSQL
# -------------------------------------------------------------
print("\nWriting churn scores to PostgreSQL...")
best_model = xgb if xgb_auc >= lr_auc else lr
X_all      = features[FEATURES].fillna(0)

features["churn_probability"] = best_model.predict_proba(X_all)[:, 1]
features["churn_risk"] = pd.cut(
    features["churn_probability"],
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"]
)

output = features[["customer_id", "churned", "churn_probability",
                    "churn_risk", "recency_to_cutoff", "frequency", "monetary"]].copy()
output.to_sql("churn_scores", engine, if_exists="replace", index=False)
print("Done. Table: churn_scores")
print(f"\nChurn risk distribution:\n{features['churn_risk'].value_counts()}")
