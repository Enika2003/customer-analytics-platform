import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
DB_URL = "postgresql+psycopg2://localhost/retail_analytics"
engine = create_engine(DB_URL)

print("Building RFM features...")

rfm_query = """
SELECT
    customer_id,
    MAX(date_id)                                      AS last_purchase_date,
    COUNT(DISTINCT invoice_id)                        AS frequency,
    SUM(revenue)                                      AS monetary
FROM fact_sales
GROUP BY customer_id
"""

df = pd.read_sql(rfm_query, engine)

# Recency = days since last purchase (relative to dataset max date)
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])
max_date = df["last_purchase_date"].max()
df["recency"] = (max_date - df["last_purchase_date"]).dt.days

# Keep only RFM columns
rfm = df[["customer_id", "recency", "frequency", "monetary"]].copy()
print(f"  {len(rfm):,} customers")
print(rfm.describe().round(2))

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])


print("\nFinding optimal number of clusters...")
inertias, silhouettes = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, labels))
    print(f"  k={k}  inertia={km.inertia_:.0f}  silhouette={silhouette_score(rfm_scaled, labels):.3f}")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(K_range), inertias, marker="o")
ax1.set_title("Elbow Curve"); ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")
ax2.plot(list(K_range), silhouettes, marker="o", color="orange")
ax2.set_title("Silhouette Score"); ax2.set_xlabel("k"); ax2.set_ylabel("Score")
plt.tight_layout()
plt.savefig("data/elbow_silhouette.png", dpi=150)
print("  Saved: data/elbow_silhouette.png")

#  Fit Final Model (k=4)
BEST_K = 4
print(f"\nFitting KMeans with k={BEST_K}...")
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
rfm["cluster"] = km_final.fit_predict(rfm_scaled)


summary = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
print("\nCluster Centroids:")
print(summary.round(2))

# Rank clusters: low recency + high freq + high monetary = Champions
summary["score"] = (
    summary["monetary"].rank() +
    summary["frequency"].rank() +
    (1 / (summary["recency"] + 1)).rank()
)
rank_order = summary["score"].rank(ascending=False).astype(int)

label_map = {
    1: "Champions",
    2: "Loyal Customers",
    3: "At Risk",
    4: "Lost"
}
rfm["segment"] = rfm["cluster"].map(rank_order).map(label_map)

print("\nSegment distribution:")
print(rfm["segment"].value_counts())


print("\nWriting segments to PostgreSQL...")
rfm.to_sql("customer_segments", engine, if_exists="replace", index=False)
print("Done. Table: customer_segments")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = ["recency", "frequency", "monetary"]
titles  = ["Recency (days)", "Frequency (orders)", "Monetary (£)"]

for ax, metric, title in zip(axes, metrics, titles):
    rfm.groupby("segment")[metric].mean().sort_values().plot(
        kind="barh", ax=ax, color=["#2196F3","#4CAF50","#FF9800","#F44336"]
    )
    ax.set_title(title)
    ax.set_xlabel("Mean Value")

plt.suptitle("Customer Segments — RFM Means", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("data/segment_profiles.png", dpi=150)
print("Saved: data/segment_profiles.png")
