WITH clicks_daily AS (
  SELECT
    date(click_ts) AS click_date,
    SUM(CASE WHEN channel = 'Google'   THEN ad_cost ELSE 0 END) AS spend_google,
    SUM(CASE WHEN channel = 'Meta'     THEN ad_cost ELSE 0 END) AS spend_meta,
    SUM(CASE WHEN channel = 'LinkedIn' THEN ad_cost ELSE 0 END) AS spend_linkedin,
    COUNTIF(channel = 'Google')   AS clicks_google,
    COUNTIF(channel = 'Meta')     AS clicks_meta,
    COUNTIF(channel = 'LinkedIn') AS clicks_linkedin
  FROM thomas_db.main.marketing_clicks
  GROUP BY 1
),

revenue_daily AS (
  SELECT
    DATE(order_ts) AS purchase_date,
    SUM(revenue)   AS revenue_total
  FROM thomas_db.main.orders
  GROUP BY 1
),

final_daily AS (
  SELECT
    strftime(c.click_date, '%Y-%m-%d') as click_date,
    r.revenue_total,
    -- spend
    c.spend_google,
    c.spend_meta,
    c.spend_linkedin, 
    -- clicks
    c.clicks_google,
    c.clicks_meta,
    c.clicks_linkedIn,
    -- controls
    e.holiday_flag,
    e.promo_discount,
    e.competitor_index
  FROM clicks_daily c
  LEFT JOIN revenue_daily r ON r.purchase_date = c.click_date
  LEFT JOIN thomas_db.main.external_vars e ON e.date = c.click_date
)


SELECT
*
FROM final_daily
ORDER BY 1;
