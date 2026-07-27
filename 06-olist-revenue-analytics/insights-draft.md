# Core Insight — Draft (source material for Phase 6/8)

## The finding
Electronics sales in Goiás (GO) have a freight-cost-to-revenue ratio of
**67.60%**, more than double the category's national average of **29.46%**.
Backed by 32 delivered orders (meets minimum sample-size threshold for
reliability — filtered out categories/regions with <30 orders as noise).

Electronics is already one of the higher-freight-ratio categories nationally
(4th highest, per 03_freight_profitability.sql query 1), and the effect is
regionally concentrated: elevated ratios also appear in Espírito Santo
(47.48%) and Pernambuco (46.20%) — states farther from Brazil's main seller
hubs (concentrated in SP/PR/MG based on regional order volume).

## Why this happens (hypothesis, not yet fully verified)
Electronics tend to be heavier/bulkier relative to their price point than
categories like bed_bath_table or housewares. Combined with longer shipping
distances to less-central states, freight cost scales up disproportionately
relative to the item's revenue.

## The business recommendation
Electronics sales in Goiás are effectively a logistics loss leader — nearly
2 out of every 3 reais of revenue in that category/region combination is
consumed by shipping. Options worth exploring (framed as recommendations,
not certainties):
- Renegotiate carrier rates for electronics shipments to GO/ES/PE specifically
- Explore local/regional fulfillment partners to cut shipping distance
- De-prioritize electronics marketing spend in these regions until freight
  economics improve, or absorb the cost consciously as a strategic trade-off

## Caveats to state honestly on the dashboard
- Freight-to-revenue ratio is a PROXY for profitability, not true margin
  (no COGS data available in this dataset)
- Sample size for GO/electronics is 32 orders — real but not huge; worth
  showing the order count alongside the ratio so viewers can judge confidence
  themselves, not just the percentage in isolation
