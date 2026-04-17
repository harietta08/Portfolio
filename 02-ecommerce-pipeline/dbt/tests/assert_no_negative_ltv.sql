-- assert_no_negative_ltv.sql
-- Custom dbt test: fails if any customer has a negative LTV value.
-- A negative LTV would indicate a data pipeline bug — likely a sign error
-- in the line_total calculation or a bad join in Silver.
-- This test runs on every dbt test invocation and blocks deployment if it fails.

select
    customer_id,
    ltv_simple,
    ltv_projected_12m
from {{ ref('customer_ltv') }}
where ltv_simple < 0
   or ltv_projected_12m < 0

-- dbt tests pass when this query returns 0 rows.
-- Any rows returned = test failure = pipeline blocked.
