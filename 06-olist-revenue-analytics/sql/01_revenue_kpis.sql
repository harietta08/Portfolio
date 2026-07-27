-- ============================================================
-- Revenue KPIs
-- ============================================================
-- Business definitions (confirmed in Phase 2 kickoff):
--   - Revenue = price only (freight tracked separately as a cost
--     metric, not blended into revenue — see 03_freight_and_profitability.sql)
--   - Only orders with order_status = 'delivered' count toward revenue
--     (undelivered/canceled orders never generated realized value)
-- ============================================================

-- ------------------------------------------------------------
-- 1. Total revenue, order count, AOV (all-time)
-- ------------------------------------------------------------
-- AOV = Average Order Value = total revenue / distinct order count
-- Note: order_items has multiple rows per order (one per line item),
-- so we sum price at the item level, then divide by DISTINCT order_id
-- count to get a true per-order average, not a per-item average.

SELECT
    ROUND(SUM(oi.price)::numeric, 2)              AS total_revenue,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2) AS avg_order_value
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered';


-- ------------------------------------------------------------
-- 2. Monthly revenue trend (basis for MoM growth)
-- ------------------------------------------------------------
-- Grouped by purchase month (not delivery month) — revenue is
-- recognized at time of purchase, delivery timing is a separate
-- operational concern (see 04_delivery_delay.sql)

SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS order_month,
    ROUND(SUM(oi.price)::numeric, 2)                       AS monthly_revenue,
    COUNT(DISTINCT o.order_id)                              AS monthly_orders,
    ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2)   AS monthly_aov
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY DATE_TRUNC('month', o.order_purchase_timestamp)
ORDER BY order_month;


-- ------------------------------------------------------------
-- 3. Month-over-Month (MoM) revenue growth %
-- ------------------------------------------------------------
-- Uses LAG() window function to compare each month to the prior month
--
-- DATA FINDING (documented, not hidden): Sept-Dec 2016 was Olist's
-- platform pilot phase — order volume was 1, 265, 0, and 1 orders
-- respectively, vs. hundreds/thousands per month from Jan 2017 onward.
-- Growth % off a base of 1-11 reais produces meaningless figures
-- (e.g. 1,025,573% MoM growth in Jan 2017). This is exactly the kind
-- of anomaly that belongs on the Executive Summary trend line as an
-- annotation — but it should NOT be included in growth-rate math.
-- Growth queries below start from 2017-01 onward for this reason;
-- the raw monthly trend query above intentionally includes all months
-- so the pilot blip is still visible on the chart.

WITH monthly AS (
    SELECT
        DATE_TRUNC('month', o.order_purchase_timestamp)::date AS order_month,
        SUM(oi.price) AS monthly_revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp >= '2017-01-01'   -- excludes 2016 pilot period
    GROUP BY DATE_TRUNC('month', o.order_purchase_timestamp)
)
SELECT
    order_month,
    ROUND(monthly_revenue::numeric, 2) AS monthly_revenue,
    ROUND(LAG(monthly_revenue) OVER (ORDER BY order_month)::numeric, 2) AS prior_month_revenue,
    ROUND(
        ((monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY order_month))
        / NULLIF(LAG(monthly_revenue) OVER (ORDER BY order_month), 0)) * 100,
        2
    ) AS mom_growth_pct
FROM monthly
ORDER BY order_month;


-- ------------------------------------------------------------
-- 4. Year-over-Year (YoY) revenue growth %
-- ------------------------------------------------------------
-- Compares each month to the same month in the prior year.
-- Same pilot-period exclusion applies here. First meaningful YoY
-- comparison is Jan 2018 vs Jan 2017 (the first full prior-year pair
-- once 2016 pilot data is excluded).

WITH monthly AS (
    SELECT
        DATE_TRUNC('month', o.order_purchase_timestamp)::date AS order_month,
        SUM(oi.price) AS monthly_revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp >= '2017-01-01'   -- excludes 2016 pilot period
    GROUP BY DATE_TRUNC('month', o.order_purchase_timestamp)
)
SELECT
    order_month,
    ROUND(monthly_revenue::numeric, 2) AS monthly_revenue,
    ROUND(LAG(monthly_revenue, 12) OVER (ORDER BY order_month)::numeric, 2) AS same_month_last_year,
    ROUND(
        ((monthly_revenue - LAG(monthly_revenue, 12) OVER (ORDER BY order_month))
        / NULLIF(LAG(monthly_revenue, 12) OVER (ORDER BY order_month), 0)) * 100,
        2
    ) AS yoy_growth_pct
FROM monthly
ORDER BY order_month;
