-- ── dbt/models/staging/stg_funding_rounds.sql ────────────────────────────────
-- Purpose: Stage-level aggregation of funding rounds per startup
-- One row per startup with stage metadata

with startups as (
    select * from {{ ref('stg_startups') }}
),

stage_ranked as (
    select
        startup_id,
        startup_name,
        sector,
        funding_stage,
        funding_amount_usd,
        funding_amount_m,
        funding_year,
        hq_country,
        hq_city,
        revenue_stage,
        case funding_stage
            when 'Seed'     then 1
            when 'Series A' then 2
            when 'Series B' then 3
            when 'Series C' then 4
            else 0
        end                                         as stage_rank,
        case
            when funding_amount_m < 5   then 'Micro (<$5M)'
            when funding_amount_m < 15  then 'Small ($5-15M)'
            when funding_amount_m < 50  then 'Mid ($15-50M)'
            else 'Large (>$50M)'
        end                                         as deal_size_bucket,
        dbt_loaded_at

    from startups
)

select * from stage_ranked
