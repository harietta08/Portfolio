-- ============================================================
-- Delivery Delay vs. Review Score (Operational Risk)
-- ============================================================
-- Delay = order_delivered_customer_date - order_estimated_delivery_date
--   Positive = delivered LATE (worse than promised)
--   Negative/zero = delivered on-time or early
--
-- Filter rule (from Phase 1 data quality notes):
--   WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
-- This excludes the 8 inconsistent rows found in Phase 1 (status says
-- delivered but no delivery timestamp exists) — can't compute a delay
-- without an actual delivered date.
-- ============================================================

-- ------------------------------------------------------------
-- 1. Order-level delivery delay with review score (base table)
-- ------------------------------------------------------------

SELECT
    o.order_id,
    o.customer_id,
    c.customer_state,
    o.order_estimated_delivery_date,
    o.order_delivered_customer_date,
    EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date))::int AS delay_days,
    r.review_score
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN order_reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
ORDER BY delay_days DESC
LIMIT 100;


-- ------------------------------------------------------------
-- 2. Average review score by delay bucket
-- ------------------------------------------------------------
-- This is the core "delay drives dissatisfaction" evidence for the
-- Operational Risk page — shows review score doesn't just dip for
-- late orders, but likely degrades progressively with delay severity.

SELECT
    CASE
        WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN '1: On-time or early'
        WHEN EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) <= 3 THEN '2: 1-3 days late'
        WHEN EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) <= 7 THEN '3: 4-7 days late'
        WHEN EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) <= 14 THEN '4: 8-14 days late'
        ELSE '5: 15+ days late'
    END AS delay_bucket,
    COUNT(*)                                    AS order_count,
    ROUND(AVG(r.review_score)::numeric, 2)      AS avg_review_score
FROM orders o
JOIN order_reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY delay_bucket
ORDER BY delay_bucket;


-- ------------------------------------------------------------
-- 3. Late delivery rate by region (customer state)
-- ------------------------------------------------------------
-- "Late" = delivered after the estimated delivery date

SELECT
    c.customer_state AS region,
    COUNT(*)                                                              AS total_orders,
    SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_orders,
    ROUND(100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_delivery_rate_pct,
    ROUND(AVG(r.review_score)::numeric, 2)                                AS avg_review_score
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN order_reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
HAVING COUNT(*) >= 30
ORDER BY late_delivery_rate_pct DESC;


-- ------------------------------------------------------------
-- 4. Late delivery rate by seller (top offenders, min order volume)
-- ------------------------------------------------------------
-- Joins through order_items since seller is attached at the line-item
-- level, not the order level. An order with multiple sellers will be
-- counted once per seller it involves (correct: we're measuring each
-- seller's fulfillment performance, not the order as a single unit).

SELECT
    oi.seller_id,
    s.seller_state,
    COUNT(DISTINCT o.order_id)                                            AS total_orders,
    SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_orders,
    ROUND(100.0 * SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) / COUNT(DISTINCT o.order_id), 2) AS late_delivery_rate_pct,
    ROUND(AVG(r.review_score)::numeric, 2)                                AS avg_review_score
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN sellers s ON s.seller_id = oi.seller_id
LEFT JOIN order_reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY oi.seller_id, s.seller_state
HAVING COUNT(DISTINCT o.order_id) >= 30
ORDER BY late_delivery_rate_pct DESC
LIMIT 25;
