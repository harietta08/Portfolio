-- ── dbt/models/marts/sector_heatmap.sql ──────────────────────────────────────
-- Purpose: Sector x stage matrix for Looker Studio heatmap
-- Powers the sector heatmap dashboard tile

with funding_rounds as (
    select * from {{ ref('stg_funding_rounds') }}
),

sector_stage as (
    select
        sector,
        funding_stage,
        stage_rank,
        deal_size_bucket,
        hq_country,
        count(*)                                    as deal_count,
        sum(funding_amount_m)                       as total_funding_m,
        avg(funding_amount_m)                       as avg_deal_size_m,
        median(funding_amount_m)                    as median_deal_size_m,
        countif(hq_country = 'USA')                 as us_deal_count,
        countif(revenue_stage = 'Revenue')          as revenue_count,
        countif(revenue_stage = 'Pre-revenue')      as pre_revenue_count,
        safe_divide(
            countif(revenue_stage = 'Revenue'),
            count(*)
        ) * 100                                     as pct_revenue_stage

    from funding_rounds
    group by 1, 2, 3, 4, 5
),

sector_totals as (
    select
        sector,
        sum(deal_count)                             as sector_total_deals,
        sum(total_funding_m)                        as sector_total_funding_m

    from sector_stage
    group by 1
)

select
    s.sector,
    s.funding_stage,
    s.stage_rank,
    s.deal_size_bucket,
    s.hq_country,
    s.deal_count,
    round(s.total_funding_m, 2)                     as total_funding_m,
    round(s.avg_deal_size_m, 2)                     as avg_deal_size_m,
    round(s.median_deal_size_m, 2)                  as median_deal_size_m,
    s.us_deal_count,
    s.revenue_count,
    s.pre_revenue_count,
    round(s.pct_revenue_stage, 1)                   as pct_revenue_stage,
    t.sector_total_deals,
    round(t.sector_total_funding_m, 2)              as sector_total_funding_m,
    round(
        safe_divide(s.deal_count, t.sector_total_deals) * 100, 1
    )                                               as pct_of_sector_deals

from sector_stage s
left join sector_totals t using (sector)
order by t.sector_total_funding_m desc, s.stage_rank
