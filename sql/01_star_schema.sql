-- =============================================================
-- Star Schema for UCI Online Retail II
-- Run this in order against the retail_analytics database
-- =============================================================

-- -------------------------------------------------------------
-- 1. DIMENSION: dim_date
-- -------------------------------------------------------------
DROP TABLE IF EXISTS dim_date CASCADE;
CREATE TABLE dim_date AS
SELECT DISTINCT
    DATE(invoicedate)                          AS date_id,
    EXTRACT(YEAR  FROM invoicedate)::INT       AS year,
    EXTRACT(MONTH FROM invoicedate)::INT       AS month,
    TO_CHAR(invoicedate, 'Month')              AS month_name,
    EXTRACT(QUARTER FROM invoicedate)::INT     AS quarter,
    EXTRACT(DOW FROM invoicedate)::INT         AS day_of_week,
    TO_CHAR(invoicedate, 'Day')                AS day_name,
    CASE WHEN EXTRACT(DOW FROM invoicedate) IN (0,6)
         THEN TRUE ELSE FALSE END              AS is_weekend
FROM raw_transactions;

ALTER TABLE dim_date ADD PRIMARY KEY (date_id);

-- -------------------------------------------------------------
-- 2. DIMENSION: dim_customer
-- -------------------------------------------------------------
DROP TABLE IF EXISTS dim_customer CASCADE;
CREATE TABLE dim_customer AS
SELECT DISTINCT ON (customer_id)
    customer_id,
    country AS primary_country
FROM (
    SELECT customer_id, country, COUNT(*) AS cnt
    FROM raw_transactions
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id, country
) sub
ORDER BY customer_id, cnt DESC;

ALTER TABLE dim_customer ADD PRIMARY KEY (customer_id);

-- -------------------------------------------------------------
-- 3. DIMENSION: dim_product
-- -------------------------------------------------------------
DROP TABLE IF EXISTS dim_product CASCADE;
CREATE TABLE dim_product AS
SELECT
    stockcode,
    -- Pick the most frequent description for each stockcode
    (SELECT description
     FROM raw_transactions r2
     WHERE r2.stockcode = r.stockcode
     GROUP BY description
     ORDER BY COUNT(*) DESC
     LIMIT 1) AS description,
    ROUND(AVG(price)::NUMERIC, 2) AS avg_unit_price
FROM raw_transactions r
GROUP BY stockcode;

ALTER TABLE dim_product ADD PRIMARY KEY (stockcode);

-- -------------------------------------------------------------
-- 4. DIMENSION: dim_geography
-- -------------------------------------------------------------
DROP TABLE IF EXISTS dim_geography CASCADE;
CREATE TABLE dim_geography AS
SELECT DISTINCT
    country,
    CASE
        WHEN country = 'United Kingdom' THEN 'Europe'
        WHEN country IN ('Germany','France','Spain','Netherlands',
                         'Belgium','Switzerland','Portugal','Italy',
                         'Sweden','Norway','Denmark','Finland',
                         'Austria','Greece','Poland','Czech Republic',
                         'Malta','Cyprus','Lithuania','Iceland') THEN 'Europe'
        WHEN country IN ('USA','Canada') THEN 'North America'
        WHEN country IN ('Australia') THEN 'Oceania'
        WHEN country IN ('Japan','Singapore','Bahrain','Lebanon',
                         'Saudi Arabia','United Arab Emirates','Israel') THEN 'Asia/Middle East'
        ELSE 'Other'
    END AS region
FROM raw_transactions;

ALTER TABLE dim_geography ADD PRIMARY KEY (country);

-- -------------------------------------------------------------
-- 5. FACT: fact_sales
-- -------------------------------------------------------------
DROP TABLE IF EXISTS fact_sales CASCADE;
CREATE TABLE fact_sales AS
SELECT
    invoice                         AS invoice_id,
    stockcode,
    customer_id,
    DATE(invoicedate)               AS date_id,
    country,
    quantity,
    price                           AS unit_price,
    revenue
FROM raw_transactions;

-- Add indexes for join performance
CREATE INDEX idx_fact_date     ON fact_sales (date_id);
CREATE INDEX idx_fact_customer ON fact_sales (customer_id);
CREATE INDEX idx_fact_product  ON fact_sales (stockcode);
CREATE INDEX idx_fact_country  ON fact_sales (country);

-- -------------------------------------------------------------
-- 6. VERIFICATION
-- -------------------------------------------------------------
SELECT 'dim_date'      AS table_name, COUNT(*) AS rows FROM dim_date
UNION ALL
SELECT 'dim_customer',                COUNT(*)          FROM dim_customer
UNION ALL
SELECT 'dim_product',                 COUNT(*)          FROM dim_product
UNION ALL
SELECT 'dim_geography',               COUNT(*)          FROM dim_geography
UNION ALL
SELECT 'fact_sales',                  COUNT(*)          FROM fact_sales;
