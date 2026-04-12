-- ── dbt/models/staging/stg_startups.sql ──────────────────────────────────────
-- Purpose: Clean and type-cast raw startup records from BigQuery
-- Source: raw startup data loaded by ingestion pipeline via GCS
-- Materialized as view — always reflects latest raw data

with source as (
    select * from {{ source('vc_raw', 'startups_raw') }}
),

cleaned as (
    select
        cast(id as int64)                           as startup_id,
        trim(name)                                  as startup_name,
        trim(description)                           as description,
        trim(sector)                                as sector,
        trim(stage)                                 as funding_stage,
        cast(funding_amount_usd as float64)         as funding_amount_usd,
        funding_amount_usd / 1000000.0              as funding_amount_m,
        cast(funding_year as int64)                 as funding_year,
        trim(hq_country)                            as hq_country,
        trim(hq_city)                               as hq_city,
        cast(founded_year as int64)                 as founded_year,
        cast(employee_count as int64)               as employee_count,
        trim(revenue_stage)                         as revenue_stage,
        trim(investors)                             as investors,
        trim(traction_signal)                       as traction_signal,
        current_timestamp()                         as dbt_loaded_at

    from source
    where
        name is not null
        and description is not null
        and funding_amount_usd > 0
        and funding_year between 2010 and 2026
)

select * from cleaned
