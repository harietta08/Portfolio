-- ============================================================
-- Freight Cost as % of Order Value (Profitability Proxy)
-- ============================================================
-- Since Olist data has no direct cost/margin field, freight-cost-
-- as-%-of-price is used as a profitability PROXY: categories/regions
-- where freight eats a large share of the order value are effectively
-- "less profitable" from an operations standpoint (higher logistics
-- cost per real of goods sold), even if we can't see true COGS.
--
-- freight_ratio = freight_value / price, at the line-item level,
-- then aggregated. Computed per line item first (not on summed
-- totals) so a single very cheap/expensive item doesn't distort
-- the ratio at the aggregate level.
-- ============================================================

-- ------------------------------------------------------------
-- 1. Freight ratio by category
-- ------------------------------------------------------------

SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
    ROUND(SUM(oi.price)::numeric, 2)                          AS total_revenue,
    ROUND(SUM(oi.freight_value)::numeric, 2)                  AS total_freight,
    COUNT(DISTINCT o.order_id)                                  AS total_orders,
    ROUND(100.0 * SUM(oi.freight_value) / NULLIF(SUM(oi.price), 0), 2) AS freight_pct_of_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
HAVING COUNT(DISTINCT o.order_id) >= 30   -- exclude tiny categories where ratio is noisy
ORDER BY freight_pct_of_revenue DESC;


-- ------------------------------------------------------------
-- 2. Freight ratio by region (customer state)
-- ------------------------------------------------------------
-- This is expected to show real geographic effect: farther/less
-- accessible states likely have higher freight-to-price ratios
-- due to longer shipping distances from concentrated seller hubs
-- (mostly SP/PR/MG based on seller distribution).

SELECT
    c.customer_state                                          AS region,
    ROUND(SUM(oi.price)::numeric, 2)                          AS total_revenue,
    ROUND(SUM(oi.freight_value)::numeric, 2)                  AS total_freight,
    COUNT(DISTINCT o.order_id)                                  AS total_orders,
    ROUND(100.0 * SUM(oi.freight_value) / NULLIF(SUM(oi.price), 0), 2) AS freight_pct_of_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY freight_pct_of_revenue DESC;


-- ------------------------------------------------------------
-- 3. Region x Category freight ratio (find the disproportionate outlier)
-- ------------------------------------------------------------
-- This is the query most likely to surface the "specific, non-obvious
-- finding" the project needs — e.g. a particular category that's
-- fine on average nationally, but has a wildly disproportionate
-- freight ratio in one specific region (small/heavy items shipped
-- long distances). HAVING clause filters out combinations with too
-- few orders to be a reliable signal (avoids reporting noise as insight).

SELECT
    c.customer_state AS region,
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
    ROUND(SUM(oi.price)::numeric, 2)   AS total_revenue,
    ROUND(SUM(oi.freight_value)::numeric, 2) AS total_freight,
    COUNT(DISTINCT o.order_id)          AS total_orders,
    ROUND(100.0 * SUM(oi.freight_value) / NULLIF(SUM(oi.price), 0), 2) AS freight_pct_of_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state, COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
HAVING COUNT(DISTINCT o.order_id) >= 30   -- minimum sample size to trust the ratio
ORDER BY freight_pct_of_revenue DESC
LIMIT 25;
