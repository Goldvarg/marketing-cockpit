-- ============================================================================
-- Marketing Mix Model Data Preparation
-- ============================================================================
-- This SQL script demonstrates how you would aggregate raw marketing data
-- from a data warehouse (BigQuery, Snowflake, Redshift) into the format
-- needed for MMM analysis.
--
-- In production, you would run this against your actual marketing tables.
-- For this project, we use Python-generated dummy data that follows this schema.
-- ============================================================================

-- Step 1: Create the aggregated daily marketing data table
CREATE TABLE IF NOT EXISTS mmm_daily_data (
    date DATE PRIMARY KEY,

    -- Channel spend (in dollars)
    spend_google DECIMAL(12,2),
    spend_meta DECIMAL(12,2),
    spend_linkedin DECIMAL(12,2),
    spend_tv DECIMAL(12,2),
    spend_email DECIMAL(12,2),

    -- Channel impressions
    impressions_google INT,
    impressions_meta INT,
    impressions_linkedin INT,
    impressions_tv INT,
    impressions_email INT,

    -- Revenue (target variable)
    revenue DECIMAL(12,2),

    -- External control variables
    is_holiday BOOLEAN,
    is_promo BOOLEAN,
    seasonality_index DECIMAL(5,2),
    competitor_spend_index DECIMAL(5,2)
);

-- Step 2: Aggregate spend data from raw marketing platform exports
-- (This is what you'd run against your actual data warehouse)
INSERT INTO mmm_daily_data (
    date,
    spend_google, spend_meta, spend_linkedin, spend_tv, spend_email,
    impressions_google, impressions_meta, impressions_linkedin, impressions_tv, impressions_email,
    revenue,
    is_holiday, is_promo, seasonality_index, competitor_spend_index
)
SELECT
    spend_date as date,

    -- Aggregate spend by channel
    SUM(CASE WHEN channel = 'Google' THEN spend ELSE 0 END) as spend_google,
    SUM(CASE WHEN channel = 'Meta' THEN spend ELSE 0 END) as spend_meta,
    SUM(CASE WHEN channel = 'LinkedIn' THEN spend ELSE 0 END) as spend_linkedin,
    SUM(CASE WHEN channel = 'TV' THEN spend ELSE 0 END) as spend_tv,
    SUM(CASE WHEN channel = 'Email' THEN spend ELSE 0 END) as spend_email,

    -- Aggregate impressions by channel
    SUM(CASE WHEN channel = 'Google' THEN impressions ELSE 0 END) as impressions_google,
    SUM(CASE WHEN channel = 'Meta' THEN impressions ELSE 0 END) as impressions_meta,
    SUM(CASE WHEN channel = 'LinkedIn' THEN impressions ELSE 0 END) as impressions_linkedin,
    SUM(CASE WHEN channel = 'TV' THEN impressions ELSE 0 END) as impressions_tv,
    SUM(CASE WHEN channel = 'Email' THEN impressions ELSE 0 END) as impressions_email,

    -- Get daily revenue from orders table
    r.daily_revenue as revenue,

    -- External variables from calendar/events table
    e.is_holiday,
    e.is_promo,
    e.seasonality_index,
    e.competitor_spend_index

FROM raw_marketing_spend s
LEFT JOIN (
    SELECT DATE(order_date) as order_date, SUM(order_value) as daily_revenue
    FROM orders
    GROUP BY DATE(order_date)
) r ON s.spend_date = r.order_date
LEFT JOIN external_events e ON s.spend_date = e.event_date
GROUP BY spend_date, r.daily_revenue, e.is_holiday, e.is_promo,
         e.seasonality_index, e.competitor_spend_index
ORDER BY date;

-- Step 3: Calculate derived metrics for analysis
-- These can be useful for data exploration and model validation
CREATE VIEW mmm_summary AS
SELECT
    DATE_TRUNC('month', date) as month,

    -- Total monthly spend by channel
    SUM(spend_google) as total_google,
    SUM(spend_meta) as total_meta,
    SUM(spend_linkedin) as total_linkedin,
    SUM(spend_tv) as total_tv,
    SUM(spend_email) as total_email,

    -- Total spend
    SUM(spend_google + spend_meta + spend_linkedin + spend_tv + spend_email) as total_spend,

    -- Total revenue
    SUM(revenue) as total_revenue,

    -- Simple ROAS by channel (not causal, just descriptive)
    SUM(revenue) / NULLIF(SUM(spend_google + spend_meta + spend_linkedin + spend_tv + spend_email), 0) as overall_roas

FROM mmm_daily_data
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

-- Step 4: Data quality checks
-- Run these before modeling to ensure data integrity

-- Check for missing dates
SELECT 'Missing dates' as check_name,
       COUNT(*) as issue_count
FROM (
    SELECT generate_series(
        (SELECT MIN(date) FROM mmm_daily_data),
        (SELECT MAX(date) FROM mmm_daily_data),
        '1 day'::interval
    )::date as expected_date
) dates
LEFT JOIN mmm_daily_data m ON dates.expected_date = m.date
WHERE m.date IS NULL;

-- Check for negative values
SELECT 'Negative spend values' as check_name,
       COUNT(*) as issue_count
FROM mmm_daily_data
WHERE spend_google < 0 OR spend_meta < 0 OR spend_linkedin < 0
   OR spend_tv < 0 OR spend_email < 0;

-- Check for outliers (> 3 std from mean)
SELECT 'Revenue outliers' as check_name,
       COUNT(*) as issue_count
FROM mmm_daily_data
WHERE revenue > (SELECT AVG(revenue) + 3 * STDDEV(revenue) FROM mmm_daily_data)
   OR revenue < (SELECT AVG(revenue) - 3 * STDDEV(revenue) FROM mmm_daily_data);
