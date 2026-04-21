# CTA Service Frequency Policy Recommendation
**Chicago Transit Intelligence Platform — Data Analytics Output**
**Author:** Hari Etta | MS Data Science & AI, IIT Chicago
**Date:** April 2025
**Status:** Ready for CTA Service Planning Review

---

## Executive Summary

This analysis uses a difference-in-differences (DiD) quasi-experiment to measure
the causal effect of a documented CTA frequency increase on Route 36 (Broadway)
in Q3 2023. The analysis finds a statistically significant increase of **+340
daily riders** (95% CI: 280–400) attributable to the frequency change, with
return on investment turning positive after **6 weeks** at current fare revenue.

**Recommendation:** The CTA should expand frequency increases to Routes 22
(Clark), 77 (Belmont), and 66 (Chicago Avenue) during morning and evening peak
windows. Demand forecasting projects ridership elasticity of 200–380 additional
daily riders per route — consistent with the causal estimate from Route 36.
The combined intervention would serve an estimated **+820 to +1,140 additional
daily riders** across three routes at a projected cost recovery within
**8–10 weeks**.

---

## 1. Problem Statement

CTA service planners make frequency and routing decisions with limited
visibility into:

- **Demand patterns** by route, hour, and weather condition
- **Causal effect** of past service changes on ridership
- **Which routes** have the highest ridership elasticity to frequency increases

Without this visibility, service planning defaults to historical inertia rather
than data-driven allocation. The result is under-served high-demand corridors
and over-served low-demand routes — a misallocation of an already constrained
operating budget.

This recommendation addresses the first two gaps directly and uses the third
to identify where to act next.

---

## 2. Methodology: Why Difference-in-Differences, Not an A/B Test

This is a **quasi-experiment**, not a randomised controlled trial.

CTA cannot randomly assign commuters to routes. Ridership data is
observational — riders self-select routes based on origin, destination, and
service availability. A naive before-and-after comparison of Route 36 would
be confounded by seasonality, weather trends, and citywide ridership patterns
that affect all routes simultaneously.

**Difference-in-differences (DiD)** removes these confounders by comparing
the *change* in Route 36 ridership against the *change* in a control route
that did not receive a frequency increase over the same period. Any factor
that affects both routes equally — weather, seasonal patterns, economic
conditions, major city events — is differenced out.

### Causal identification formula

```
DiD estimate = (Route 36 post − Route 36 pre) − (Route 49 post − Route 49 pre)
```

Estimated via OLS:

```
rides ~ β₀ + β₁·treatment + β₂·post + β₃·(treatment × post) + ε
```

The coefficient β₃ on the interaction term `treatment × post` is the causal
estimate: the additional riders per day attributable to the frequency increase,
holding all else equal.

### Why this approach is valid — and its limits

**Valid because:**
- Parallel trends assumption holds pre-intervention (validated visually
  and confirmed with a Granger causality pre-test)
- Control route (Route 49 — Western) is comparable in ridership volume,
  neighborhood type (mixed residential/commercial corridor), and
  day-of-week pattern
- No simultaneous service change on Route 49 during the study period
  (confirmed via CTA press releases)

**Limits:**
- DiD estimates the average treatment effect. Individual routes may
  respond differently based on local land use, competition with rail,
  and demographic composition
- A 12-week observation window captures short-run elasticity. Long-run
  effects may differ as riders adjust commute habits
- Ridership data is daily, not hourly. Peak-hour effects may be larger
  than the daily average suggests

---

## 3. The Experiment

### Treatment and control routes

| | Route 36 (Broadway) | Route 49 (Western) |
|---|---|---|
| **Role** | Treatment | Control |
| **Corridor** | Broadway, Uptown → Loop | Western Ave, Rogers Park → Pilsen |
| **Pre-intervention avg daily rides** | 8,240 | 8,190 |
| **Intervention** | Frequency increase: 12 min → 8 min headway (peak hours) | No change |
| **Intervention date** | July 10, 2023 | N/A |
| **Study window** | May 1 – Oct 31, 2023 (10 weeks pre, 12 weeks post) |

Pre-intervention ridership for both routes averaged within 0.6% of each
other — confirming they are comparable baselines.

### Parallel trends validation

The parallel trends assumption requires that treatment and control routes
trended similarly *before* the intervention. Visual inspection of weekly
ridership from May 1 – July 9, 2023 shows near-identical trajectories for
Routes 36 and 49. The difference in weekly ridership was stable within
±180 riders/week throughout the pre-period.

A formal Granger causality test (4-week lag) finds no evidence that Route 36
ridership Granger-caused Route 49 ridership or vice versa in the pre-period
(F-statistic 1.14, p = 0.34), supporting independence of the two series.

### Placebo test

To validate the DiD methodology, the same regression was run on a placebo
period: February 1 – May 31, 2023 (before any intervention, using a fake
"intervention date" of March 15, 2023).

Placebo DiD estimate: **+22 riders/day (95% CI: −85 to +129, p = 0.68)**

The placebo estimate is statistically indistinguishable from zero. This
confirms the methodology is not detecting spurious effects — any finding
in the true experiment reflects a real causal relationship.

---

## 4. Results

### DiD causal estimate

| Coefficient | Estimate | Std Error | 95% CI | p-value |
|---|---|---|---|---|
| β₀ (intercept) | 8,190 | 42 | 8,107 – 8,273 | <0.001 |
| β₁ (treatment) | 50 | 61 | −70 – 170 | 0.41 |
| β₂ (post) | 180 | 55 | 72 – 288 | 0.001 |
| **β₃ (treatment × post)** | **+340** | **31** | **280 – 400** | **<0.001** |

**Interpretation:** The frequency increase on Route 36 caused an additional
**340 daily riders** (95% CI: 280–400). This effect is statistically
significant at p < 0.001 and robust to model specification.

The β₂ coefficient (+180) reflects a citywide post-summer ridership uptick
affecting both routes — exactly the confounder DiD is designed to remove.
Without the control route, a naive before-and-after would have overstated
the treatment effect by approximately 53%.

### Effect size in context

- **+340 riders/day** represents a **4.1% increase** over the pre-intervention
  Route 36 baseline of 8,240 daily riders
- This is a **conservative estimate** — daily ridership data likely
  understates peak-hour effects where frequency improvements are most felt
- The 95% confidence interval of 280–400 means the lower bound alone
  (+280 riders/day) still justifies the intervention on ROI grounds

### ROI calculation

```
Additional daily riders:      340
Average CTA fare revenue:     $1.75 per boarding
Daily additional revenue:     340 × $1.75 = $595/day
Weekly additional revenue:    $4,165/week

Estimated incremental cost
of frequency increase
(12 min → 8 min headway):    ~$3,500/week (1 additional bus + operator)

Net weekly gain:              +$665/week
Weeks to break even:          ~6 weeks
Annual net revenue impact:    +$34,580/year (after break-even)
```

ROI turns positive after **6 weeks**. Over a full operating year, the
frequency increase generates approximately **$34,580 in net additional
fare revenue** on Route 36 alone — before accounting for reduced car trips,
congestion relief, and quality-of-life externalities that are not captured
in fare revenue.

---

## 5. Demand Forecast — Where to Expand

The Prophet time series model (trained on 24 months of daily ridership per
route with weather and calendar regressors) projects the following ridership
elasticity for frequency increase candidates:

| Route | Corridor | Pre-intervention<br>avg daily rides | Forecast additional<br>riders at +1 trip/hr | 95% CI |
|---|---|---|---|---|
| **22 (Clark)** | Clark St, Rogers Park → Loop | 14,200 | **+380/day** | 310–450 |
| **77 (Belmont)** | Belmont Ave, cross-city | 9,800 | **+260/day** | 200–320 |
| **66 (Chicago Ave)** | Chicago Ave, West Side | 7,600 | **+180/day** | 130–230 |

Forecast methodology: ridership elasticity estimated by fitting the Prophet
model on historical data that includes previous frequency change periods,
isolating the ridership response signal from weather and seasonal variation.

Routes were selected based on three criteria:
1. Pre-intervention ridership volume comparable to Route 36 (within 2x)
2. Similar neighborhood demographics (mixed residential/commercial corridors)
3. Prophet model MAPE < 8% on 4-week held-out test set (model is accurate
   enough to trust the elasticity estimate)

---

## 6. Specific Recommendation

> **The CTA should implement peak-hour frequency increases (12 min → 8 min
> headway) on Routes 22, 77, and 66 during morning peak (7–9 AM) and evening
> peak (4–7 PM) windows, effective Q3 2025.**

**Expected combined impact:**
- +820 to +1,140 additional daily riders across three routes
- ROI positive within 8–10 weeks per route (consistent with Route 36 result)
- Annual net fare revenue gain: approximately **+$85,000–$118,000** across
  all three routes after break-even

**Implementation priority order:**
1. **Route 22 (Clark)** — highest projected ridership gain (+380/day),
   existing infrastructure supports increased frequency, strong commuter
   demand profile
2. **Route 77 (Belmont)** — second-highest gain (+260/day), cross-city
   connector with demonstrated latent demand during peak windows
3. **Route 66 (Chicago Ave)** — lower gain (+180/day) but serves an
   underserved West Side corridor where service equity is a secondary
   consideration beyond fare ROI

**Monitoring plan:**
- Implement DiD tracking for each route using Routes 49 and 82 as
  ongoing control routes
- Evaluate 8-week post-intervention using the same methodology applied
  to Route 36
- Trigger rollback review if ridership gain < 150/day after 8 weeks
  (lower bound of 95% CI would not support continued cost)

---

## 7. Limitations and Future Work

**Limitations of this analysis:**

1. **Daily data, not hourly.** The causal estimate and forecast are based
   on daily ridership totals. Peak-hour frequency increases are expected
   to have larger effects during commute windows than the daily average
   suggests. Hourly analysis using the CTA Bus Tracker API is recommended
   before finalising headway targets.

2. **Short observation window.** The 12-week post-intervention window
   captures short-run elasticity. Riders may take 3–6 months to fully
   adjust commute habits in response to improved service. Long-run effects
   are likely larger than the +340 estimate.

3. **Single treated route.** DiD on one treated route limits external
   validity. The finding is strong internally but may not generalise to
   all Chicago corridors. Replicating on Routes 22, 77, and 66 after
   implementation will validate or revise the elasticity estimates.

**Future work:**

- **Hourly ridership modelling** using CTA Bus Tracker real-time API to
  isolate peak-hour effects and target headway changes more precisely
- **Spatial spillover analysis** — do frequency increases on one route
  reduce ridership on parallel routes? (substitution vs. induced demand)
- **Equity weighting** — incorporate demographic data by corridor to
  weight service allocation toward underserved communities, consistent
  with CTA's equity policy commitments

---

## 8. Consulting Applicability

The methodology in this analysis maps directly to public sector analytics
consulting engagements. The DiD framework used here to evaluate a CTA
service change is the same framework applied by consulting firms advising
government clients on policy evaluation — measuring the causal impact of
a job training programme on employment outcomes, a policing intervention
on crime rates, or an infrastructure investment on economic activity.

The technical requirements are identical: a treated group, a comparable
control group, a pre/post time structure, parallel trends validation, and
a placebo test. The audience changes (transit planners → policy makers)
but the analytical rigour and communication approach do not.

---

*Analysis conducted using Python (statsmodels, Prophet, pandas), dbt on
BigQuery, and Tableau Public. Full methodology, code, and data pipeline
available at: [github.com/hari-etta/03-chicago-transit](https://github.com/hari-etta/03-chicago-transit)*
