import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
DB_URL = "postgresql+psycopg2://localhost/retail_analytics"
engine = create_engine(DB_URL)


# 1. Temporal Split
#    First 18 months → features
#    Last 6 months   → actual future revenue (CLV target)
print("Loading transactions...")
query = "SELECT customer_id, date_id, invoice_id, revenue FROM fact_sales"
df = pd.read_sql(query, engine)
df["date_id"] = pd.to_datetime(df["date_id"])

min_date = df["date_id"].min()
max_date = df["date_id"].max()
cutoff   = min_date + pd.Timedelta(days=540)   # ~18 months for features

print(f"  Dataset range  : {min_date.date()} → {max_date.date()}")
print(f"  Feature period : {min_date.date()} → {cutoff.date()}")
print(f"  Target period  : {cutoff.date()} → {max_date.date()}")

hist   = df[df["date_id"] <  cutoff]
future = df[df["date_id"] >= cutoff]

# 2. Build Features (historical behaviour)
print("\nBuilding features...")
features = hist.groupby("customer_id").agg(
    frequency       = ("invoice_id", "nunique"),
    total_revenue   = ("revenue",    "sum"),
    avg_order_value = ("revenue",    "mean"),
    min_order_value = ("revenue",    "min"),
    max_order_value = ("revenue",    "max"),
    last_purchase   = ("date_id",    "max"),
    first_purchase  = ("date_id",    "min"),
).reset_index()

features["recency"]       = (cutoff - features["last_purchase"]).dt.days
features["tenure_days"]   = (features["last_purchase"] - features["first_purchase"]).dt.days
features["avg_days_between"] = (
    features["tenure_days"] / features["frequency"]
).replace([np.inf, np.nan], 0)

# 3. Build Target: actual revenue in future period
future_rev = future.groupby("customer_id")["revenue"].sum().reset_index()
future_rev.columns = ["customer_id", "future_revenue"]

data = features.merge(future_rev, on="customer_id", how="left")
data["future_revenue"] = data["future_revenue"].fillna(0)

# Only model customers who appeared in both periods
data = data[data["future_revenue"] > 0].copy()
print(f"  Customers with future revenue: {len(data):,}")
print(f"  Avg future revenue: £{data['future_revenue'].mean():,.2f}")
print(f"  Max future revenue: £{data['future_revenue'].max():,.2f}")

# Log-transform target (revenue is heavily right-skewed)
data["log_future_revenue"] = np.log1p(data["future_revenue"])

# 4. Train / Test Split
FEATURES = ["recency", "frequency", "total_revenue", "avg_order_value",
            "min_order_value", "max_order_value", "tenure_days", "avg_days_between"]

X = data[FEATURES].fillna(0)
y = data["log_future_revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Train Model
print("\nTraining CLV model (GradientBoosting)...")
model = GradientBoostingRegressor(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
y_pred_real = np.expm1(y_pred)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred_real)
r2  = r2_score(y_test, y_pred)

print(f"  R²  (log scale) : {r2:.3f}")
print(f"  MAE (£)         : £{mae:,.2f}")

cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"  Cross-val R²    : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

# 6. Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Actual vs Predicted
axes[0].scatter(y_test_real, y_pred_real, alpha=0.4, color="#2196F3", s=15)
max_val = max(y_test_real.max(), y_pred_real.max())
axes[0].plot([0, max_val], [0, max_val], "r--", lw=1)
axes[0].set_xlabel("Actual CLV (£)")
axes[0].set_ylabel("Predicted CLV (£)")
axes[0].set_title(f"Actual vs Predicted CLV\nR²={r2:.3f}  MAE=£{mae:,.0f}")
axes[0].set_xlim(0, np.percentile(y_test_real, 99))
axes[0].set_ylim(0, np.percentile(y_pred_real, 99))

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
importances.plot(kind="barh", ax=axes[1], color="#4CAF50")
axes[1].set_title("Feature Importance")

# CLV distribution
all_pred = np.expm1(model.predict(X))
axes[2].hist(all_pred, bins=50, color="#FF9800", edgecolor="white")
axes[2].set_xlabel("Predicted CLV (£)")
axes[2].set_ylabel("Number of Customers")
axes[2].set_title("CLV Distribution")
axes[2].set_xlim(0, np.percentile(all_pred, 95))

plt.tight_layout()
plt.savefig("data/clv_model.png", dpi=150)
print("Saved: data/clv_model.png")
print("\nWriting CLV scores to PostgreSQL...")
all_features = features.copy()
X_all        = all_features[FEATURES].fillna(0)

all_features["predicted_clv"] = np.expm1(model.predict(X_all))
all_features["clv_segment"] = pd.qcut(
    all_features["predicted_clv"],
    q=4,
    labels=["Bronze", "Silver", "Gold", "Platinum"]
)

output = all_features[["customer_id", "predicted_clv", "clv_segment",
                         "frequency", "total_revenue", "recency"]].copy()
output.to_sql("clv_scores", engine, if_exists="replace", index=False)
print("Done. Table: clv_scores")

print("\nCLV segment distribution:")
print(all_features["clv_segment"].value_counts().sort_index())
print(f"\nAvg predicted CLV by segment:")
print(all_features.groupby("clv_segment")["predicted_clv"].mean().apply(lambda x: f"£{x:,.2f}"))
