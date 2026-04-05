library(ggplot2)
library(dplyr)
library(RPostgres)
library(scales)
library(patchwork)
library(viridis)

# --- Connection ---
con <- dbConnect(RPostgres::Postgres(),
                 dbname   = "retail_analytics",
                 host     = "localhost",
                 port     = 5432,
                 user     = Sys.getenv("USER"))

theme_clean <- theme_minimal(base_size = 13) +
  theme(plot.title    = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(colour = "grey50"),
        panel.grid.minor = element_blank())

# =============================================================
# 1. Revenue Over Time
# =============================================================
rev_time <- dbGetQuery(con, "
  SELECT
    DATE_TRUNC('month', date_id) AS month,
    SUM(revenue)                 AS monthly_revenue
  FROM fact_sales
  GROUP BY 1
  ORDER BY 1
")
rev_time$month <- as.Date(rev_time$month)

p1 <- ggplot(rev_time, aes(x = month, y = monthly_revenue)) +
  geom_col(fill = "#2196F3", alpha = 0.7) +
  geom_smooth(method = "loess", se = FALSE, colour = "#F44336", linewidth = 1) +
  scale_y_continuous(labels = label_currency(prefix = "£", big.mark = ",")) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(title    = "Monthly Revenue Trend",
       subtitle = "UCI Online Retail II — 2009 to 2011",
       x = NULL, y = "Revenue") +
  theme_clean +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# =============================================================
# 2. Customer Segment Profiles (RFM)
# =============================================================
segments <- dbGetQuery(con, "
  SELECT segment,
         AVG(recency)   AS avg_recency,
         AVG(frequency) AS avg_frequency,
         AVG(monetary)  AS avg_monetary,
         COUNT(*)       AS n_customers
  FROM customer_segments
  GROUP BY segment
")

seg_long <- tidyr::pivot_longer(
  segments,
  cols      = c(avg_recency, avg_frequency, avg_monetary),
  names_to  = "metric",
  values_to = "value"
)
seg_long$metric <- recode(seg_long$metric,
  avg_recency   = "Recency (days)",
  avg_frequency = "Frequency (orders)",
  avg_monetary  = "Monetary (£)"
)
seg_long$segment <- factor(seg_long$segment,
  levels = c("Champions", "Loyal Customers", "At Risk", "Lost"))

p2 <- ggplot(seg_long, aes(x = segment, y = value, fill = segment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_manual(values = c("#4CAF50","#2196F3","#FF9800","#F44336")) +
  labs(title    = "Customer Segment Profiles",
       subtitle = "Mean RFM values per segment",
       x = NULL, y = NULL) +
  theme_clean +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# =============================================================
# 3. Churn Risk by Country (Top 10)
# =============================================================
churn_country <- dbGetQuery(con, "
  SELECT
    dc.primary_country                    AS country,
    COUNT(*)                              AS total_customers,
    SUM(CASE WHEN cs.churn_risk = 'High'
             THEN 1 ELSE 0 END)           AS high_risk,
    ROUND(100.0 * SUM(CASE WHEN cs.churn_risk = 'High'
             THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_high_risk
  FROM churn_scores cs
  JOIN dim_customer dc USING (customer_id)
  GROUP BY dc.primary_country
  HAVING COUNT(*) >= 20
  ORDER BY pct_high_risk DESC
  LIMIT 10
")

p3 <- ggplot(churn_country,
             aes(x = reorder(country, pct_high_risk), y = pct_high_risk,
                 fill = pct_high_risk)) +
  geom_col() +
  geom_text(aes(label = paste0(pct_high_risk, "%")),
            hjust = -0.2, size = 3.5) +
  coord_flip() +
  scale_fill_viridis_c(option = "magma", direction = -1, guide = "none") +
  scale_y_continuous(limits = c(0, 105)) +
  labs(title    = "High Churn Risk by Country",
       subtitle = "Countries with ≥ 20 customers",
       x = NULL, y = "% High Risk Customers") +
  theme_clean

# =============================================================
# 4. CLV Distribution by Segment
# =============================================================
clv <- dbGetQuery(con, "SELECT clv_segment, predicted_clv FROM clv_scores")
clv$clv_segment <- factor(clv$clv_segment,
                           levels = c("Bronze","Silver","Gold","Platinum"))

p4 <- ggplot(clv, aes(x = predicted_clv, fill = clv_segment)) +
  geom_histogram(bins = 40, colour = "white", linewidth = 0.2) +
  facet_wrap(~clv_segment, scales = "free") +
  scale_x_continuous(labels = label_currency(prefix = "£", big.mark = ",")) +
  scale_fill_manual(values = c("#CD7F32","#C0C0C0","#FFD700","#E5E4E2"),
                    guide  = "none") +
  labs(title    = "Predicted CLV Distribution by Segment",
       subtitle = "Gradient Boosting model — 6-month forward revenue",
       x = "Predicted CLV (£)", y = "Customers") +
  theme_clean

# =============================================================
# 5. Save all charts
# =============================================================
ggsave("data/chart_revenue_trend.png",    p1, width = 10, height = 5, dpi = 150)
ggsave("data/chart_segment_profiles.png", p2, width = 12, height = 5, dpi = 150)
ggsave("data/chart_churn_country.png",    p3, width = 8,  height = 5, dpi = 150)
ggsave("data/chart_clv_distribution.png", p4, width = 12, height = 6, dpi = 150)

# Combined dashboard page
combined <- (p1 | p3) / (p2) / (p4)
ggsave("data/chart_dashboard.png", combined, width = 16, height = 14, dpi = 150)

cat("All charts saved to data/\n")
dbDisconnect(con)
