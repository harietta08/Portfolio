-- stg_events.sql
-- Source: events data exported from the synthetic simulation
-- Purpose: Standardize event stream columns for funnel and A/B analysis.

with source as (
    select * from {{ source('ecommerce_gold', 'events') }}
),

renamed as (
    select
        cast(event_id as STRING)           as event_id,
        cast(session_id as INT64)          as session_id,
        cast(user_id as INT64)             as user_id,
        cast(event_type as STRING)         as event_type,
        cast(device as STRING)             as device,
        cast(channel as STRING)            as channel,
        cast(variant as STRING)            as ab_variant,
        cast(product_id as INT64)          as product_id,
        cast(timestamp as TIMESTAMP)       as event_timestamp,
        date(cast(timestamp as TIMESTAMP)) as event_date

    from source
    where event_id is not null
      and session_id is not null
      and event_type in (
          'page_view', 'product_view', 'add_to_cart',
          'checkout_start', 'purchase'
      )
)

select * from renamed
