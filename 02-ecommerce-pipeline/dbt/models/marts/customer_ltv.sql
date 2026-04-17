-- customer_ltv.sql
-- Materialization: INCREMENTAL — only processes new/updated customers
-- Unique key: user_id — MERGE on user_id when re-running
-- Purpose: Final customer LTV mart. One row per customer.
--          This is the primary table powering the Looker Studio LTV tab.

{{
    config(
        materialized       = 'incremental',
        incremental_strategy = 'merge',
        unique_key         = 'customer_id',
        on_schema_change   = 'sync_all_columns'
    )
}}

with customer_journey as (
    select * from {{ ref('int_customer_journey') }}

    {% if is_incremental() %}
    -- On incremental runs: only process customers whose data changed
    -- since the last run. _gold_timestamp tracks when Gold was last updated.
    where customer_id not in (
        select customer_id from {{ this }}
        where last_order_date >= date_sub(current_date(), interval 7 day)
    )
    {% endif %}
),

ltv_tiers as (
    select
        *,
        -- LTV tier for dashboard segmentation
        case
            when ltv_projected_12m >= 1000 then 'platinum'
            when ltv_projected_12m >= 500  then 'gold'
            when ltv_projected_12m >= 200  then 'silver'
            else 'bronze'
        end as ltv_tier,

        -- Cohort month for retention analysis
        date_trunc(first_order_date, month) as cohort_month,

        -- Months since acquisition
        date_diff(current_date(), first_order_date, month) as months_since_acquisition,

        -- At-risk flag: high-value customers showing churn signals
        case
            when customer_segment = 'high_value'
             and days_since_last_order between 30 and 60
            then true
            else false
        end as is_at_risk

    from customer_journey
)

select
    customer_id,
    username,
    email,
    city,
    customer_segment,
    ltv_tier,
    is_churned,
    is_at_risk,
    favourite_category,
    preferred_device,

    -- Order metrics
    total_orders,
    total_spend,
    avg_order_value,
    total_items,
    unique_products,

    -- LTV metrics
    ltv_simple,
    ltv_projected_12m,
    order_frequency_monthly,

    -- Dates
    first_order_date,
    last_order_date,
    acquisition_date,
    cohort_month,
    days_since_last_order,
    customer_lifespan_days,
    months_since_acquisition,

    -- Session behavior
    total_sessions,
    session_conversion_rate,
    avg_session_duration_seconds,
    mobile_sessions,
    desktop_sessions,

    current_timestamp() as _mart_updated_at

from ltv_tiers
