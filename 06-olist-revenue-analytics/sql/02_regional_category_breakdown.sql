-- ============================================================
-- Regional & Category Revenue Breakdown
-- ============================================================
-- Definitions carried over from 01_revenue_kpis.sql:
--   - Revenue = price only (freight excluded, tracked separately)
--   - Only order_status = 'delivered' counts
-- "Region" = customer's state (customer_state) — i.e. where the
-- BUYER is located, not the seller. This matches a revenue/demand
-- lens: "where is our revenue coming from," not a fulfillment lens.
-- ============================================================

-- ------------------------------------------------------------
-- 1. Revenue by region (customer state)
-- ------------------------------------------------------------

SELECT
    c.customer_state                                        AS region,
    ROUND(SUM(oi.price)::numeric, 2)                        AS total_revenue,
    COUNT(DISTINCT o.order_id)                                AS total_orders,
    ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2)     AS avg_order_value,
    ROUND(100.0 * SUM(oi.price) / SUM(SUM(oi.price)) OVER (), 2) AS pct_of_total_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY total_revenue DESC;


-- ------------------------------------------------------------
-- 2. Revenue by product category (English name, with fallback)
-- ------------------------------------------------------------
-- COALESCE handles the 2 categories with no English translation
-- (pc_gamer, portateis_cozinha_e_preparadores_de_alimentos) —
-- falls back to the Portuguese name rather than dropping the rows.
-- Products with NULL category (610 rows, ~1.85%) are labeled
-- 'unknown_category' rather than excluded, so their revenue isn't
-- silently lost from the total.

SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
    ROUND(SUM(oi.price)::numeric, 2)                        AS total_revenue,
    COUNT(DISTINCT o.order_id)                                AS total_orders,
    ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2)     AS avg_order_value,
    ROUND(100.0 * SUM(oi.price) / SUM(SUM(oi.price)) OVER (), 2) AS pct_of_total_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
ORDER BY total_revenue DESC;


-- ------------------------------------------------------------
-- 3. Revenue by region AND category (cross-tab, for drill-down)
-- ------------------------------------------------------------
-- Useful for Power BI matrix visuals / slicers combining both
-- dimensions at once.

SELECT
    c.customer_state AS region,
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
    ROUND(SUM(oi.price)::numeric, 2) AS total_revenue,
    COUNT(DISTINCT o.order_id)         AS total_orders
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state, COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
ORDER BY region, total_revenue DESC;
