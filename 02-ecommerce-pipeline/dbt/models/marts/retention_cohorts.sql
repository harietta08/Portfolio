-- retention_cohorts.sql
-- Purpose: Monthly cohort retention analysis.
--          For each cohort (month of first order), calculate what % of
--          customers made a repeat purchase in each subsequent month.
--          Powers the retention grid in Looker Studio.

with customer_orders as (
    select
        order_user_id                                           as customer_id,
        first_order_date,
        date_trunc(first_order_date, month)                    as cohort_month,
        last_order_date,
        total_orders,
        total_spend,
        customer_segment,
        days_since_last_order,
        months_since_acquisition

    from {{ ref('stg_orders') }}
),

cohort_sizes as (
    select
        cohort_month,
        count(distinct customer_id) as cohort_size

    from customer_orders
    group by cohort_month
),

-- Retention: customers who ordered in month N after acquisition
retention_data as (
    select
        co.cohort_month,
        co.months_since_acquisition                           as period_number,
        count(distinct co.customer_id)                        as retained_customers,
        sum(co.total_spend)                                   as cohort_revenue,
        avg(co.total_spend)                                   as avg_spend_per_customer

    from customer_orders co
    where co.months_since_acquisition >= 0
    group by co.cohort_month, co.months_since_acquisition
),

final as (
    select
        r.cohort_month,
        r.period_number,
        r.retained_customers,
        cs.cohort_size,
        round(safe_divide(r.retained_customers, cs.cohort_size) * 100, 1)
            as retention_rate,
        round(r.cohort_revenue, 2)                            as cohort_revenue,
        round(r.avg_spend_per_customer, 2)                    as avg_spend_per_customer,
        current_timestamp()                                   as _mart_updated_at

    from retention_data r
    left join cohort_sizes cs on r.cohort_month = cs.cohort_month
)

select * from final
order by cohort_month, period_number
