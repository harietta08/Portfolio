-- ab_test_segments.sql
-- Purpose: Break A/B test results down by device and acquisition channel.
--          Checks for heterogeneous treatment effects — does Variant B work
--          better on mobile than desktop? On paid vs organic?
--          Segment breakdown is required to make a rollout recommendation.

with sessions as (
    select * from {{ ref('int_sessions') }}
),

segment_results as (
    select
        ab_variant,
        device,
        channel,
        count(distinct session_id)                              as total_sessions,
        sum(converted)                                          as converted_sessions,
        round(
            safe_divide(sum(converted), count(distinct session_id)) * 100,
            4
        )                                                       as conversion_rate,
        count(distinct user_id)                                 as unique_users,
        avg(session_duration_seconds)                           as avg_session_duration_seconds

    from sessions
    group by ab_variant, device, channel
),

-- Pivot to show A vs B side by side per segment
device_comparison as (
    select
        device,
        channel,
        max(case when ab_variant = 'A' then total_sessions end)  as variant_a_sessions,
        max(case when ab_variant = 'B' then total_sessions end)  as variant_b_sessions,
        max(case when ab_variant = 'A' then conversion_rate end) as variant_a_conversion_rate,
        max(case when ab_variant = 'B' then conversion_rate end) as variant_b_conversion_rate,
        round(
            max(case when ab_variant = 'B' then conversion_rate end) -
            max(case when ab_variant = 'A' then conversion_rate end),
            4
        )                                                        as conversion_lift_pp,
        current_timestamp()                                      as _mart_updated_at

    from segment_results
    group by device, channel
)

select * from device_comparison
order by device, channel
