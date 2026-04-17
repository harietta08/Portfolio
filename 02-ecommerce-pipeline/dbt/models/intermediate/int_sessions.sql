-- int_sessions.sql
-- Purpose: Aggregate event-level data to session level.
--          One row per session with funnel progression flags.
--          Used by ab_test_results and ab_test_segments mart models.

with events as (
    select * from {{ ref('stg_events') }}
),

session_agg as (
    select
        session_id,
        user_id,
        ab_variant,
        device,
        channel,
        date(min(event_timestamp))       as session_date,
        min(event_timestamp)             as session_start,
        max(event_timestamp)             as session_end,

        -- Funnel flags — did this session reach each stage?
        max(case when event_type = 'page_view'      then 1 else 0 end) as reached_page_view,
        max(case when event_type = 'product_view'   then 1 else 0 end) as reached_product_view,
        max(case when event_type = 'add_to_cart'    then 1 else 0 end) as reached_add_to_cart,
        max(case when event_type = 'checkout_start' then 1 else 0 end) as reached_checkout,
        max(case when event_type = 'purchase'       then 1 else 0 end) as converted,

        count(distinct event_type)       as distinct_event_types,
        count(*)                         as total_events

    from events
    group by session_id, user_id, ab_variant, device, channel
),

final as (
    select
        *,
        -- Session duration in seconds
        timestamp_diff(session_end, session_start, second) as session_duration_seconds,

        -- Furthest funnel stage reached
        case
            when converted          = 1 then 'purchase'
            when reached_checkout   = 1 then 'checkout_start'
            when reached_add_to_cart = 1 then 'add_to_cart'
            when reached_product_view = 1 then 'product_view'
            else 'page_view'
        end as furthest_stage

    from session_agg
)

select * from final
