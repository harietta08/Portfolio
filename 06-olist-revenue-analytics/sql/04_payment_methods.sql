-- ============================================================
-- Payment Method Distribution & Installment Patterns
-- ============================================================
-- Uses order_payments table directly (payment_value here reflects
-- the actual amount paid via that payment record — this can include
-- freight, unlike order_items.price, since Olist's payment records
-- represent the total transaction amount, not itemized goods cost).
-- This is a DIFFERENT revenue basis than 01/02/03 on purpose — this
-- section answers "how do customers pay," not "how much did we earn."
-- ============================================================

-- ------------------------------------------------------------
-- 1. Payment method distribution (share of orders and value)
-- ------------------------------------------------------------

SELECT
    op.payment_type,
    COUNT(DISTINCT op.order_id)                                  AS order_count,
    ROUND(100.0 * COUNT(DISTINCT op.order_id) / SUM(COUNT(DISTINCT op.order_id)) OVER (), 2) AS pct_of_orders,
    ROUND(SUM(op.payment_value)::numeric, 2)                     AS total_payment_value,
    ROUND(100.0 * SUM(op.payment_value) / SUM(SUM(op.payment_value)) OVER (), 2) AS pct_of_value,
    ROUND(AVG(op.payment_value)::numeric, 2)                     AS avg_payment_value
FROM order_payments op
JOIN orders o ON o.order_id = op.order_id
WHERE o.order_status = 'delivered'
GROUP BY op.payment_type
ORDER BY total_payment_value DESC;


-- ------------------------------------------------------------
-- 2. Installment patterns — distribution of installment counts
-- ------------------------------------------------------------
-- Only meaningful for credit_card payments (other types typically
-- use 1 installment by definition — boleto/voucher/debit are
-- single-payment methods)

SELECT
    op.payment_installments,
    COUNT(DISTINCT op.order_id)                                  AS order_count,
    ROUND(100.0 * COUNT(DISTINCT op.order_id) / SUM(COUNT(DISTINCT op.order_id)) OVER (), 2) AS pct_of_credit_card_orders,
    ROUND(AVG(op.payment_value)::numeric, 2)                     AS avg_payment_value
FROM order_payments op
JOIN orders o ON o.order_id = op.order_id
WHERE o.order_status = 'delivered'
  AND op.payment_type = 'credit_card'
GROUP BY op.payment_installments
ORDER BY op.payment_installments;


-- ------------------------------------------------------------
-- 3. Does higher order value correlate with more installments?
-- ------------------------------------------------------------
-- Buckets credit card orders by payment value range, shows avg
-- installment count per bucket — tests the intuitive hypothesis
-- that customers finance bigger purchases over more installments.

SELECT
    CASE
        WHEN op.payment_value < 50 THEN '1: Under R$50'
        WHEN op.payment_value < 100 THEN '2: R$50-99'
        WHEN op.payment_value < 250 THEN '3: R$100-249'
        WHEN op.payment_value < 500 THEN '4: R$250-499'
        WHEN op.payment_value < 1000 THEN '5: R$500-999'
        ELSE '6: R$1000+'
    END AS order_value_bucket,
    COUNT(DISTINCT op.order_id)              AS order_count,
    ROUND(AVG(op.payment_installments), 2)   AS avg_installments,
    ROUND(AVG(op.payment_value)::numeric, 2) AS avg_payment_value
FROM order_payments op
JOIN orders o ON o.order_id = op.order_id
WHERE o.order_status = 'delivered'
  AND op.payment_type = 'credit_card'
GROUP BY order_value_bucket
ORDER BY order_value_bucket;
