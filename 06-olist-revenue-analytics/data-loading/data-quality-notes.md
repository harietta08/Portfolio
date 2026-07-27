# Data Quality Notes — Phase 1

Findings from initial data quality checks on the raw Olist CSVs.
These inform filtering logic used in Phase 2 SQL queries.

## 1. `olist_geolocation_dataset.csv`
- 261,831 exact duplicate rows out of 1,000,163.
- Expected: this is a zip-code-prefix → lat/long lookup table with repeated
  entries. Will deduplicate to one row per zip prefix (avg lat/long) at load time.

## 2. `olist_order_reviews_dataset.csv`
- `review_comment_title`: 88.34% null
- `review_comment_message`: 58.7% null
- Expected: most customers leave a star rating without a written comment.
  Not used for text analysis in this project (out of scope) — only
  `review_score` (numeric) is used.

## 3. `olist_orders_dataset.csv` — delivery date nulls
- `order_approved_at`: 0.16% null
- `order_delivered_carrier_date`: 1.79% null
- `order_delivered_customer_date`: 2.98% null

Order status breakdown (all orders):
| status      | count |
|-------------|-------|
| delivered   | 96,478 |
| shipped     | 1,107 |
| canceled    | 625 |
| unavailable | 609 |
| invoiced    | 314 |
| processing  | 301 |
| created     | 5 |
| approved    | 2 |

Confirmed: the vast majority of null delivery dates belong to orders with
non-'delivered' status (shipped, canceled, unavailable, invoiced, processing,
created) — expected, since these orders never reached the customer.

**Anomaly found:** 8 orders have `order_status = 'delivered'` but a NULL
`order_delivered_customer_date`. This is a genuine inconsistency in the
source data (status says delivered, but no delivery timestamp recorded).

**Anomaly found:** 14 orders have `order_status = 'delivered'` but a NULL
`order_approved_at`.

**Rule adopted for all delivery-delay SQL/DAX logic:**
```sql
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
```
This excludes the 8 inconsistent rows from any delay-vs-review-score analysis.

## 4. `olist_products_dataset.csv`
- 610 rows (1.85%) missing `product_category_name` (and correlated fields:
  name length, description length, photo qty).
- 2 rows missing weight/dimension fields.
- Decision (to be finalized in Phase 2): label missing category as
  `'unknown_category'` rather than dropping rows, to avoid losing revenue
  from the aggregate totals.

## 5. Other tables
`olist_customers_dataset.csv`, `olist_order_items_dataset.csv`,
`olist_order_payments_dataset.csv`, `olist_sellers_dataset.csv`,
`product_category_name_translation.csv` — no nulls, no duplicate rows found.

## 6. Foreign key integrity checks (across all tables)
All core FK relationships are fully intact (0 missing references):
- `orders.customer_id` -> `customers.customer_id`
- `order_items.order_id` -> `orders.order_id`
- `order_items.product_id` -> `products.product_id`
- `order_items.seller_id` -> `sellers.seller_id`
- `order_payments.order_id` -> `orders.order_id`
- `order_reviews.order_id` -> `orders.order_id`
- No duplicate `order_id` values in the `orders` table.

**Anomaly found:** 2 `product_category_name` values in `products` have no
matching row in `product_category_name_translation`:
- `pc_gamer`
- `portateis_cozinha_e_preparadores_de_alimentos`

**Rule adopted:** use `COALESCE(english_name, portuguese_name)` when joining
to the translation table, so these products aren't dropped from category-level
revenue analysis.

**Anomaly found:** 775 orders in `orders` have zero matching rows in
`order_items` (i.e., no line items at all). Breakdown by status:
| status      | count |
|-------------|-------|
| unavailable | 603 |
| canceled    | 164 |
| created     | 5 |
| invoiced    | 2 |
| shipped     | 1 |

## 7. 2016 "pilot period" — the Executive Summary anomaly
Discovered while building revenue KPI queries (Phase 2), not Phase 1, but
documented here since it's a data characteristic, not a query bug.

Monthly order volume for Sept–Dec 2016:
| month    | orders | revenue  |
|----------|--------|----------|
| 2016-09  | 1      | R$134.97 |
| 2016-10  | 265    | R$40,325.11 |
| 2016-11  | 0      | — |
| 2016-12  | 1      | R$10.90 |

Compared to Jan 2017 onward, where volume jumps to hundreds/thousands of
orders per month and stays there. This is Olist's platform pilot/ramp-up
phase, not representative operating volume.

**Impact:** growth-rate calculations (MoM/YoY) off this tiny base produce
mathematically correct but meaningless percentages (e.g. 1,025,573% MoM
growth in Jan 2017, 6,660,754% YoY growth in Dec 2017).

**Rule adopted:** the raw monthly trend (used for the Executive Summary
chart) includes all months, since this pilot blip is a legitimate, useful
anomaly to annotate on the trend line. Growth % calculations (MoM/YoY)
filter to `order_purchase_timestamp >= '2017-01-01'` so percentages are
computed off a meaningful base. First valid YoY comparison is Jan 2018 vs
Jan 2017.
