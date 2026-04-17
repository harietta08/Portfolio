-- stg_customers.sql
-- Source: gold_customer_kpis (customer dimension attributes only)
-- Purpose: Clean customer profile data for use in joins downstream.

with source as (
    select * from {{ source('ecommerce_gold', 'gold_customer_kpis') }}
),

renamed as (
    select
        cast(user_id as INT64)             as customer_id,
        cast(username as STRING)           as username,
        cast(email as STRING)              as email,
        cast(address_city as STRING)       as city,
        cast(favourite_category as STRING) as favourite_category,
        cast(customer_segment as STRING)   as customer_segment,
        cast(is_churned as BOOL)           as is_churned,
        cast(first_order_date as DATE)     as acquisition_date,
        cast(_gold_timestamp as TIMESTAMP) as _gold_timestamp

    from source
    where user_id is not null
      and username is not null
)

select * from renamed
