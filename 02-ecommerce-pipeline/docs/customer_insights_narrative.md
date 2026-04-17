# Customer Insights Narrative
**Project:** E-Commerce Customer Analytics Pipeline  
**Author:** Hari Etta  
**Audience:** Marketing and Product teams

## Top Finding
Electronics drives 68% of total pipeline revenue despite representing only 30% of product SKUs. Average order value for electronics ($367) is 4x higher than clothing ($24). The highest-ROI business intervention is electronics cart abandonment recovery — capturing even 10% of abandoned electronics carts would increase monthly revenue by an estimated 12%.

## Customer Segmentation
Three value segments emerge from the LTV analysis:

**High-value customers (total spend > $500):** Small group, disproportionate revenue contribution. Churn risk is highest here — a 60-day recency window flags these customers for re-engagement campaigns before they fully lapse.

**Mid-value customers (spend $200–$500):** Largest segment by count. Highest growth potential. These customers show consistent purchase frequency but lower average order values, suggesting cross-sell opportunities into higher-margin categories.

**Low-value customers (spend < $200):** Primarily single-purchase customers. Focus here should be on second-purchase conversion — the data shows customers who make a second purchase within 30 days of their first have 3x higher 12-month LTV than those who don't.

## A/B Test: Checkout Simplification
**Experiment:** 2-step checkout (Variant B) vs 3-step checkout (Variant A)  
**Result:** Variant B shows a statistically significant 3pp lift in conversion rate (38% → 41%), p < 0.05, 95% CI: (1.2pp, 4.8pp).  
**Guardrails:** Average order value held steady — no revenue dilution from the higher conversion rate.  
**Recommendation:** Full rollout of Variant B. Estimated annual revenue impact: +8% on checkout-initiated sessions based on current traffic volume.

## Retention Analysis
Month-0 retention (customers who order again within their acquisition month) is 45%. This drops to 28% by month 2. Industry benchmark for e-commerce is 25–35% at month 2, putting this pipeline's customer base at the high end.

**Actionable insight for the product team:** The sharpest drop-off is between month 0 and month 1 (45% → 28%). A targeted email campaign in the 2–3 week window after first purchase — timed before the month-1 drop-off — would likely recover 5–8% of at-risk customers based on industry lift rates for triggered email.

## Channel Performance
Organic and paid search drive 55% of sessions but 72% of high-value customer acquisitions. Social and email drive higher session volume but lower LTV customers. Budget allocation recommendation: maintain paid search investment for acquisition; use email for retention and re-engagement of existing mid-value segment.

## Supply Chain Applicability
The same medallion architecture and cohort analysis patterns used here map directly to supply chain analytics. Customer LTV curves parallel demand forecasting curves. Cohort retention analysis mirrors component lifecycle analysis. The pipeline architecture would apply at companies like Caterpillar (equipment parts demand), Abbott (medical device inventory), or Boeing (supply chain risk monitoring) with dataset substitution only — no architectural changes required.
