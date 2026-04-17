-- ab_test_results.sql
-- Purpose: Aggregate A/B test results at variant level.
--          Control A = 3-step checkout, Treatment B = 2-step checkout.
--          Primary metric: session conversion rate.
--          Guardrail metrics: checked in 04_ab_test_analysis.ipynb.

with sessions as (
    select * from {{ ref('int_sessions') }}
),

variant_results as (
    select
        ab_variant,
        count(distinct session_id)                              as total_sessions,
        sum(converted)                                          as converted_sessions,
        round(
            safe_divide(sum(converted), count(distinct session_id)) * 100,
            4
        )                                                       as conversion_rate,
        count(distinct user_id)                                 as unique_users,
        avg(session_duration_seconds)                           as avg_session_duration_seconds,

        -- Funnel breakdown
        sum(reached_product_view)                               as reached_product_view,
        sum(reached_add_to_cart)                                as reached_add_to_cart,
        sum(reached_checkout)                                   as reached_checkout,

        min(session_date)                                       as experiment_start_date,
        max(session_date)                                       as experiment_end_date

    from sessions
    group by ab_variant
),

final as (
    select
        ab_variant,
        total_sessions,
        converted_sessions,
        conversion_rate,
        unique_users,
        round(avg_session_duration_seconds, 1)                  as avg_session_duration_seconds,
        reached_product_view,
        reached_add_to_cart,
        reached_checkout,
        round(safe_divide(reached_product_view, total_sessions) * 100, 2)
            as product_view_rate,
        round(safe_divide(reached_add_to_cart, total_sessions) * 100, 2)
            as add_to_cart_rate,
        round(safe_divide(reached_checkout, total_sessions) * 100, 2)
            as checkout_rate,
        experiment_start_date,
        experiment_end_date,
        current_timestamp() as _mart_updated_at

    from variant_results
)

select * from final
order by ab_variant
