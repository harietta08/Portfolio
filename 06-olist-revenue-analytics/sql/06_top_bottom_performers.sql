-- ============================================================
-- Top/Bottom Performing Categories & Sellers
-- ============================================================
-- "Margin proxy" = 100 - freight_pct_of_revenue (i.e. the inverse of
-- the freight ratio from 03_freight_profitability.sql). Higher =
-- less of each revenue real consumed by shipping. Still a PROXY,
-- not true margin — no COGS data available in this dataset.
-- Minimum order thresholds applied throughout to avoid small-sample
-- noise masquerading as a "top" or "bottom" performer.
-- ============================================================

-- ------------------------------------------------------------
-- 1. Top 10 categories by revenue
-- ------------------------------------------------------------

SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
    ROUND(SUM(oi.price)::numeric, 2) AS total_revenue,
    COUNT(DISTINCT o.order_id)         AS total_orders
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
ORDER BY total_revenue DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 2. Bottom 10 categories by revenue (min 30 orders, avoid noise)
-- ------------------------------------------------------------

SELECT
    COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
    ROUND(SUM(oi.price)::numeric, 2) AS total_revenue,
    COUNT(DISTINCT o.order_id)         AS total_orders
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
HAVING COUNT(DISTINCT o.order_id) >= 30
ORDER BY total_revenue ASC
LIMIT 10;


-- ------------------------------------------------------------
-- 3. Top/Bottom 10 categories by margin proxy (min 30 orders)
-- ------------------------------------------------------------

WITH category_margin AS (
    SELECT
        COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category') AS category,
        ROUND(SUM(oi.price)::numeric, 2)         AS total_revenue,
        COUNT(DISTINCT o.order_id)                 AS total_orders,
        ROUND(100 - (100.0 * SUM(oi.freight_value) / NULLIF(SUM(oi.price), 0)), 2) AS margin_proxy_pct
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN products p ON p.product_id = oi.product_id
    LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
    WHERE o.order_status = 'delivered'
    GROUP BY COALESCE(ct.product_category_name_english, p.product_category_name, 'unknown_category')
    HAVING COUNT(DISTINCT o.order_id) >= 30
)
-- Top 10 by margin proxy
(SELECT 'top' AS rank_type, category, total_revenue, total_orders, margin_proxy_pct
 FROM category_margin
 ORDER BY margin_proxy_pct DESC
 LIMIT 10)
UNION ALL
-- Bottom 10 by margin proxy
(SELECT 'bottom' AS rank_type, category, total_revenue, total_orders, margin_proxy_pct
 FROM category_margin
 ORDER BY margin_proxy_pct ASC
 LIMIT 10)
ORDER BY rank_type, margin_proxy_pct DESC;


-- ------------------------------------------------------------
-- 4. Top 10 sellers by revenue
-- ------------------------------------------------------------

SELECT
    oi.seller_id,
    s.seller_state,
    ROUND(SUM(oi.price)::numeric, 2) AS total_revenue,
    COUNT(DISTINCT o.order_id)         AS total_orders
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN sellers s ON s.seller_id = oi.seller_id
WHERE o.order_status = 'delivered'
GROUP BY oi.seller_id, s.seller_state
ORDER BY total_revenue DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 5. Bottom 10 sellers by revenue (min 10 orders, avoid one-off noise)
-- ------------------------------------------------------------

SELECT
    oi.seller_id,
    s.seller_state,
    ROUND(SUM(oi.price)::numeric, 2) AS total_revenue,
    COUNT(DISTINCT o.order_id)         AS total_orders
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN sellers s ON s.seller_id = oi.seller_id
WHERE o.order_status = 'delivered'
GROUP BY oi.seller_id, s.seller_state
HAVING COUNT(DISTINCT o.order_id) >= 10
ORDER BY total_revenue ASC
LIMIT 10;


-- ------------------------------------------------------------
-- 6. Top/Bottom 10 sellers by margin proxy (min 30 orders)
-- ------------------------------------------------------------

WITH seller_margin AS (
    SELECT
        oi.seller_id,
        s.seller_state,
        ROUND(SUM(oi.price)::numeric, 2)         AS total_revenue,
        COUNT(DISTINCT o.order_id)                 AS total_orders,
        ROUND(100 - (100.0 * SUM(oi.freight_value) / NULLIF(SUM(oi.price), 0)), 2) AS margin_proxy_pct
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN sellers s ON s.seller_id = oi.seller_id
    WHERE o.order_status = 'delivered'
    GROUP BY oi.seller_id, s.seller_state
    HAVING COUNT(DISTINCT o.order_id) >= 30
)
(SELECT 'top' AS rank_type, seller_id, seller_state, total_revenue, total_orders, margin_proxy_pct
 FROM seller_margin
 ORDER BY margin_proxy_pct DESC
 LIMIT 10)
UNION ALL
(SELECT 'bottom' AS rank_type, seller_id, seller_state, total_revenue, total_orders, margin_proxy_pct
 FROM seller_margin
 ORDER BY margin_proxy_pct ASC
 LIMIT 10)
ORDER BY rank_type, margin_proxy_pct DESC;
