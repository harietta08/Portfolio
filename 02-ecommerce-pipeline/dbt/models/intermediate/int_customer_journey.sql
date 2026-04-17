-- int_customer_journey.sql
-- Purpose: Build a complete customer journey combining order history
--          and session behavior. One row per customer.
--          Used by customer_ltv and retention_cohorts mart models.

with customers as (
    select * from {{ ref('stg_customers') }}
),

orders as (
    select * from {{ ref('stg_orders') }}
),

sessions as (
    select
        user_id,
        count(distinct session_id)                          as total_sessions,
        sum(converted)                                      as total_converted_sessions,
        avg(session_duration_seconds)                       as avg_session_duration_seconds,
        count(distinct case when device = 'mobile'
              then session_id end)                          as mobile_sessions,
        count(distinct case when device = 'desktop'
              then session_id end)                          as desktop_sessions,
        count(distinct channel)                             as unique_channels,
        max(session_date)                                   as last_session_date

    from {{ ref('int_sessions') }}
    group by user_id
),

joined as (
    select
        c.customer_id,
        c.username,
        c.email,
        c.city,
        c.favourite_category,
        c.customer_segment,
        c.is_churned,
        c.acquisition_date,

        -- Order metrics
        o.total_orders,
        o.total_spend,
        o.avg_order_value,
        o.total_items,
        o.unique_products,
        o.first_order_date,
        o.last_order_date,
        o.days_since_last_order,
        o.customer_lifespan_days,
        o.order_frequency_monthly,
        o.ltv_simple,
        o.ltv_projected_12m,

        -- Session metrics
        coalesce(s.total_sessions, 0)              as total_sessions,
        coalesce(s.total_converted_sessions, 0)    as total_converted_sessions,
        coalesce(s.avg_session_duration_seconds, 0) as avg_session_duration_seconds,
        coalesce(s.mobile_sessions, 0)             as mobile_sessions,
        coalesce(s.desktop_sessions, 0)            as desktop_sessions,
        coalesce(s.unique_channels, 0)             as unique_channels,

        -- Derived
        case
            when coalesce(s.total_sessions, 0) > 0
            then round(safe_divide(
                coalesce(s.total_converted_sessions, 0),
                s.total_sessions
            ) * 100, 2)
            else 0
        end as session_conversion_rate,

        -- Preferred device
        case
            when coalesce(s.mobile_sessions, 0) > coalesce(s.desktop_sessions, 0)
            then 'mobile'
            else 'desktop'
        end as preferred_device

    from customers c
    left join orders  o on c.customer_id = o.order_user_id
    left join sessions s on c.customer_id = s.user_id
)

select * from joined
