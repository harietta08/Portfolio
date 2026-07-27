-- ============================================================
-- Olist Revenue & Operations Analytics — Schema Definition
-- ============================================================
-- Constraints reflect data quality findings from Phase 1:
-- - FK integrity confirmed clean across all core tables
-- - NOT NULL only applied where nulls were confirmed absent or
--   would break the row's meaning (e.g. price, order_status)
-- - Nullable left as-is for known legitimate nulls (delivery
--   dates, review comments, product_category_name)
-- - geolocation table has no PK (raw lookup, deduplicated at
--   load time, not enforced at schema level)
-- ============================================================

-- Drop tables if re-running this script (clean slate)
DROP TABLE IF EXISTS order_reviews CASCADE;
DROP TABLE IF EXISTS order_payments CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS sellers CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS category_translation CASCADE;
DROP TABLE IF EXISTS geolocation CASCADE;

-- ============================================================
-- Lookup tables (no dependencies)
-- ============================================================

CREATE TABLE category_translation (
    product_category_name          VARCHAR(100) PRIMARY KEY,
    product_category_name_english  VARCHAR(100)
);

CREATE TABLE geolocation (
    geolocation_zip_code_prefix  VARCHAR(10),
    geolocation_lat              NUMERIC(10, 6),
    geolocation_lng               NUMERIC(10, 6),
    geolocation_city              VARCHAR(100),
    geolocation_state             VARCHAR(2)
);

-- ============================================================
-- Entity tables
-- ============================================================

CREATE TABLE customers (
    customer_id                VARCHAR(32) PRIMARY KEY,
    customer_unique_id         VARCHAR(32) NOT NULL,
    customer_zip_code_prefix   VARCHAR(10),
    customer_city              VARCHAR(100),
    customer_state              VARCHAR(2)
);

CREATE TABLE sellers (
    seller_id                VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix   VARCHAR(10),
    seller_city               VARCHAR(100),
    seller_state              VARCHAR(2)
);

CREATE TABLE products (
    product_id                    VARCHAR(32) PRIMARY KEY,
    product_category_name         VARCHAR(100) REFERENCES category_translation(product_category_name),
    product_name_lenght            NUMERIC,
    product_description_lenght      NUMERIC,
    product_photos_qty             NUMERIC,
    product_weight_g               NUMERIC,
    product_length_cm              NUMERIC,
    product_height_cm              NUMERIC,
    product_width_cm               NUMERIC
);

-- ============================================================
-- Transactional tables
-- ============================================================

CREATE TABLE orders (
    order_id                        VARCHAR(32) PRIMARY KEY,
    customer_id                     VARCHAR(32) NOT NULL REFERENCES customers(customer_id),
    order_status                    VARCHAR(20) NOT NULL,
    order_purchase_timestamp        TIMESTAMP NOT NULL,
    order_approved_at               TIMESTAMP,
    order_delivered_carrier_date    TIMESTAMP,
    order_delivered_customer_date   TIMESTAMP,
    order_estimated_delivery_date   TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    order_id             VARCHAR(32) NOT NULL REFERENCES orders(order_id),
    order_item_id        INTEGER NOT NULL,
    product_id           VARCHAR(32) NOT NULL REFERENCES products(product_id),
    seller_id            VARCHAR(32) NOT NULL REFERENCES sellers(seller_id),
    shipping_limit_date  TIMESTAMP,
    price                NUMERIC(10, 2) NOT NULL,
    freight_value        NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE order_payments (
    order_id              VARCHAR(32) NOT NULL REFERENCES orders(order_id),
    payment_sequential    INTEGER NOT NULL,
    payment_type          VARCHAR(20) NOT NULL,
    payment_installments  INTEGER NOT NULL,
    payment_value          NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE order_reviews (
    review_id                 VARCHAR(32) NOT NULL,
    order_id                  VARCHAR(32) NOT NULL REFERENCES orders(order_id),
    review_score              INTEGER NOT NULL,
    review_comment_title      VARCHAR(200),
    review_comment_message    TEXT,
    review_creation_date      TIMESTAMP NOT NULL,
    review_answer_timestamp   TIMESTAMP NOT NULL,
    PRIMARY KEY (review_id, order_id)
);

-- Note on order_reviews PK: using a composite key (review_id, order_id)
-- rather than review_id alone, since we haven't specifically verified
-- review_id is unique on its own — this is the safer choice until confirmed.
-- We'll verify this assumption during the load step below.
