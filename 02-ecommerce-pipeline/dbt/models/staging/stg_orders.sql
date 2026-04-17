-- stg_orders.sql
-- Source: gold_customer_kpis exported from Databricks to BigQuery
-- Purpose: Standardize column names, cast types, add surrogate keys.
--          Staging models never join — one source table in, clean rows out.

with source as (
    select * from {{ source('ecommerce_gold', 'gold_customer_kpis') }}
),

renamed as (
    select
        -- Primary key
        cast(user_id as INT64)          as order_user_id,

        -- Order metrics
        cast(total_orders as INT64)     as total_orders,
        cast(total_spend as FLOAT64)    as total_spend,
        cast(avg_order_value as FLOAT64) as avg_order_value,
        cast(total_items as INT64)      as total_items,
        cast(unique_products as INT64)  as unique_products,

        -- Dates
        cast(first_order_date as DATE)  as first_order_date,
        cast(last_order_date as DATE)   as last_order_date,

        -- Recency
        cast(days_since_last_order as INT64)     as days_since_last_order,
        cast(customer_lifespan_days as INT64)    as customer_lifespan_days,
        cast(order_frequency_monthly as FLOAT64) as order_frequency_monthly,

        -- LTV
        cast(ltv_simple as FLOAT64)        as ltv_simple,
        cast(ltv_projected_12m as FLOAT64) as ltv_projected_12m,

        -- Segment
        cast(customer_segment as STRING)   as customer_segment,
        cast(is_churned as BOOL)           as is_churned,
        cast(favourite_category as STRING) as favourite_category,

        -- User info
        cast(username as STRING)           as username,
        cast(email as STRING)              as email,
        cast(address_city as STRING)       as address_city,

        -- Audit
        cast(_gold_timestamp as TIMESTAMP) as _gold_timestamp,
        cast(_as_of_date as DATE)          as _as_of_date

    from source
    where user_id is not null
      and total_spend >= 0
)

select * from renamed
