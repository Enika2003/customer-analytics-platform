# Customer Analytics Platform

I built this project to understand what it actually looks like when a data team works end-to-end on a customer analytics problem — not just the modelling part, but the full picture: getting data into a database, cleaning it properly, building a warehouse, running ML, producing charts worth showing someone, and building a dashboard a non-technical person could use.

The dataset is real transaction data from a UK-based online retailer (UCI Online Retail II — about 1 million rows spanning 2009 to 2011). The stack is PostgreSQL, Python, R, and Tableau.

---

## What I built and why

### Getting the data in

The raw dataset comes as an Excel file with two sheets (one per year). The first thing I noticed was how messy it actually was — about 240,000 rows with no customer ID, another 18,000 cancellation invoices (these start with "C"), and a handful of rows with negative quantities or zero prices. I wrote a Python ingestion script that handles all of this before anything touches the database.

After cleaning: **805,549 rows**, **5,878 customers**, **4,631 products**, **41 countries**.

### Modelling it properly (star schema)

Loading a flat file into one big table is fine for exploration, but it's not how real data teams work. I modelled the data into a proper star schema with a central fact table and four dimension tables.

```
fact_sales (805,549 rows)
    ├── dim_customer   (5,878 customers)
    ├── dim_product    (4,631 products)
    ├── dim_date       (604 dates)
    └── dim_geography  (41 countries)
```

The trickiest part was `dim_customer` — some customers had placed orders from multiple countries, so I had to pick one primary country per customer using a frequency-based approach rather than just `DISTINCT`.

### Customer segmentation (K-Means on RFM)

I built RFM features — Recency (days since last purchase), Frequency (number of orders), and Monetary (total spend) — and ran K-Means clustering. I tested k=2 through k=8 using both inertia and silhouette scores to find the right number of clusters.

k=4 gave the most interpretable business segments:

| Segment | Customers | Avg Recency | Avg Orders | Avg Spend |
|---|---|---|---|---|
| Champions | 4 | 3 days | 213 orders | £436k |
| Loyal Customers | 38 | 24 days | 100 orders | £79k |
| At Risk | 3,839 | 67 days | 7 orders | £2,984 |
| Lost | 1,997 | 463 days | 2 orders | £763 |

The Champions and Loyal groups are clearly wholesale buyers — the order volumes and spend levels are too high for individual consumers. That said, they represent a tiny slice of the customer base (42 out of 5,878) but almost certainly drive a disproportionate share of revenue. That's a real business insight worth acting on.

### Churn prediction (Logistic Regression + XGBoost)

My first attempt at churn prediction gave an AUC of 1.0 — which immediately told me something was wrong. I had used recency as a feature while also defining churn as "recency ≥ 180 days". The model was just memorising the label. Classic data leakage.

I rebuilt it properly using a **temporal split**: build features from the first 21 months, define churn based on whether the customer purchased in the final 3 months. This way the model never sees the future when learning.

The real results:
- **Logistic Regression AUC: 0.802**
- **XGBoost AUC: 0.814**
- 56.6% of customers churned in the observation window
- 2,333 customers flagged as high churn risk

An AUC of 0.81 is a genuinely useful model. The most important features were recency to cutoff, frequency, and average days between orders — which makes intuitive sense.

### CLV forecasting (Gradient Boosting)

I framed CLV as a regression problem: given a customer's historical behaviour up to a cutoff date, predict how much they'll spend in the next 6 months.

Revenue distributions are heavily right-skewed (a handful of customers spend 100x the average), so I log-transformed the target before training and exponentiated predictions back to £ values.

- **R²: 0.54** (cross-validated: 0.50 ± 0.035)
- **MAE: £1,707**

The R² isn't spectacular — CLV forecasting on transaction data is genuinely hard, especially when a single wholesale customer can skew everything. But 0.54 is honest and defensible, and the cross-validation consistency (±0.035) tells me the model isn't overfitting.

Customers are grouped into four tiers:

| Tier | Avg Predicted 6-Month CLV |
|---|---|
| Platinum | £2,687 |
| Gold | £575 |
| Silver | £370 |
| Bronze | £223 |

### Visualisations (R / ggplot2)

I used R's ggplot2 to produce four charts:

- **Monthly revenue trend** — with a LOESS smoothing line. Revenue clearly peaks every Q4 (October–November), consistent across both years.
- **Segment profiles** — RFM means per cluster, showing how different the Champions are from the Lost group.
- **Churn risk by country** — stacked bars showing the proportion of high/medium/low churn risk customers by country, filtered to countries with at least 20 customers.
- **CLV distribution by tier** — histograms per segment showing the spread within each tier.

### Dashboard (Tableau)

I connected Tableau directly to PostgreSQL using a custom SQL query that joins the fact table with all ML output tables. The dashboard shows:

- Monthly revenue trend
- Customer segment breakdown
- Churn risk by country

![Dashboard](data/dashboard.png)

---

## Key findings

**Revenue is seasonal.** Both years show the same Q4 spike. Any retention or acquisition campaign should account for this — October is not the time to run experiments, it's the time to convert.

**Most customers are at risk or lost.** 3,839 customers sit in the At Risk bucket. These are people who bought at least once but haven't been retained. The business question this raises: are they one-time buyers by nature, or is there a retention problem?

**A tiny group of wholesale buyers drives outsized revenue.** The 42 Champions and Loyal customers almost certainly account for a significant portion of total revenue. Losing one of them hurts far more than losing 100 regular customers.

**Churn is predictable but not perfectly.** An AUC of 0.81 means the model is genuinely useful for prioritising who to target with a retention campaign — but it's not certain. The 24% false positive rate means some "at risk" customers would have stayed anyway.

---

## How to run it yourself

### Requirements
- Python 3.12+
- PostgreSQL 17+
- R 4.4+

### Setup

```bash
# Python dependencies
pip install pandas sqlalchemy psycopg2-binary openpyxl scikit-learn xgboost matplotlib seaborn

# R packages (run inside R)
install.packages(c("ggplot2", "dplyr", "RPostgres", "scales", "patchwork", "viridis"))

# Create the database
psql postgres -c "CREATE DATABASE retail_analytics;"
```

### Run the pipeline

```bash
# 1. Load raw data into PostgreSQL
python3 scripts/ingest.py

# 2. Build the star schema
psql retail_analytics -f sql/01_star_schema.sql

# 3. Run ML models
python3 scripts/ml_segmentation.py
python3 scripts/ml_churn.py
python3 scripts/ml_clv.py

# 4. Generate ggplot2 charts
Rscript scripts/visualisations.R
```

---

## Repository structure

```
customer-analytics/
├── data/
│   ├── online_retail_II.xlsx
│   ├── dashboard.png
│   ├── chart_revenue_trend.png
│   ├── chart_segment_profiles.png
│   ├── chart_churn_country.png
│   ├── chart_clv_distribution.png
│   └── elbow_silhouette.png
├── scripts/
│   ├── ingest.py
│   ├── ml_segmentation.py
│   ├── ml_churn.py
│   ├── ml_clv.py
│   └── visualisations.R
├── sql/
│   └── 01_star_schema.sql
└── README.md
```
