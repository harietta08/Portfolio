-- ── dbt/models/marts/funding_trends.sql ──────────────────────────────────────
-- Purpose: Aggregated funding trends by sector and year
-- Powers Looker Studio time-series charts
-- Materialized as table — pre-aggregated for dashboard performance

with funding_rounds as (
    select * from {{ ref('stg_funding_rounds') }}
),

sector_year as (
    select
        sector,
        funding_year,
        funding_stage,
        count(*)                                    as deal_count,
        sum(funding_amount_usd)                     as total_funding_usd,
        sum(funding_amount_m)                       as total_funding_m,
        avg(funding_amount_m)                       as avg_deal_size_m,
        median(funding_amount_m)                    as median_deal_size_m,
        min(funding_amount_m)                       as min_deal_size_m,
        max(funding_amount_m)                       as max_deal_size_m,
        countif(revenue_stage = 'Revenue')          as revenue_stage_count,
        countif(revenue_stage = 'Pre-revenue')      as pre_revenue_count

    from funding_rounds
    group by 1, 2, 3
),

with_growth as (
    select
        *,
        lag(total_funding_m) over (
            partition by sector
            order by funding_year
        )                                           as prev_year_funding_m,

        safe_divide(
            total_funding_m - lag(total_funding_m) over (
                partition by sector order by funding_year
            ),
            lag(total_funding_m) over (
                partition by sector order by funding_year
            )
        ) * 100                                     as yoy_growth_pct

    from sector_year
)

select
    sector,
    funding_year,
    funding_stage,
    deal_count,
    round(total_funding_m, 2)                       as total_funding_m,
    round(avg_deal_size_m, 2)                       as avg_deal_size_m,
    round(median_deal_size_m, 2)                    as median_deal_size_m,
    revenue_stage_count,
    pre_revenue_count,
    round(yoy_growth_pct, 1)                        as yoy_growth_pct

from with_growth
order by sector, funding_year, funding_stage
