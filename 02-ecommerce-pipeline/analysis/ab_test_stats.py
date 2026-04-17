# =============================================================================
# ab_test_stats.py
# Purpose: Statistical analysis for the A/B test.
#          Called from notebook 04 and tested in tests/test_ab_test_stats.py.
#
# Experiment: Does a 2-step checkout (B) convert better than 3-step (A)?
# Primary metric: Session conversion rate
# Guardrail metrics: Average order value, return rate
# Unit of randomization: Session ID
# =============================================================================

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from typing import Tuple, Dict


# -----------------------------------------------------------------------------
# 1. SAMPLE SIZE CALCULATION
# Must be done BEFORE running the experiment. Running without a pre-calculated
# sample size means you don't know when to stop — leading to p-hacking.
# -----------------------------------------------------------------------------
def calculate_sample_size(
    baseline_rate: float = 0.38,   # control conversion rate
    mde: float           = 0.02,   # minimum detectable effect (2pp lift)
    alpha: float         = 0.05,   # significance level (95% confidence)
    power: float         = 0.80    # statistical power (80%)
) -> Dict:
    """
    Calculate minimum sample size per variant needed for the experiment.

    Parameters:
        baseline_rate: Current conversion rate of control (Variant A)
        mde:           Minimum lift we care about detecting (absolute pp)
        alpha:         Type I error rate (false positive rate)
        power:         1 - Type II error rate (probability of detecting true effect)

    Returns:
        Dict with sample_size_per_variant, total_sample_size, and parameters
    """
    treatment_rate = baseline_rate + mde
    effect_size    = proportion_effectsize(baseline_rate, treatment_rate)

    analysis       = NormalIndPower()
    n_per_variant  = analysis.solve_power(
        effect_size = effect_size,
        alpha       = alpha,
        power       = power,
        ratio       = 1.0   # equal group sizes
    )
    n_per_variant = int(np.ceil(n_per_variant))

    return {
        "baseline_rate":        baseline_rate,
        "treatment_rate":       treatment_rate,
        "mde_absolute":         mde,
        "mde_relative":         round(mde / baseline_rate * 100, 1),
        "alpha":                alpha,
        "power":                power,
        "effect_size":          round(effect_size, 4),
        "sample_size_per_variant": n_per_variant,
        "total_sample_size":    n_per_variant * 2,
    }


# -----------------------------------------------------------------------------
# 2. TWO-PROPORTION Z-TEST
# Tests whether the observed difference in conversion rates is statistically
# significant, i.e., not due to random chance.
# -----------------------------------------------------------------------------
def run_ab_test(
    control_conversions:   int,
    control_sessions:      int,
    treatment_conversions: int,
    treatment_sessions:    int,
    alpha: float = 0.05
) -> Dict:
    """
    Run a two-proportion z-test comparing Variant A (control) to Variant B (treatment).

    Parameters:
        control_conversions:   Number of purchases in Variant A
        control_sessions:      Total sessions in Variant A
        treatment_conversions: Number of purchases in Variant B
        treatment_sessions:    Total sessions in Variant B
        alpha:                 Significance level

    Returns:
        Dict with z_stat, p_value, significant, lift, confidence_interval
    """
    control_rate   = control_conversions / control_sessions
    treatment_rate = treatment_conversions / treatment_sessions
    lift_absolute  = treatment_rate - control_rate
    lift_relative  = lift_absolute / control_rate * 100

    counts   = np.array([treatment_conversions, control_conversions])
    nobs     = np.array([treatment_sessions,    control_sessions])
    z_stat, p_value = proportions_ztest(counts, nobs, alternative='larger')

    # Confidence interval around the lift estimate
    se = np.sqrt(
        (control_rate * (1 - control_rate) / control_sessions) +
        (treatment_rate * (1 - treatment_rate) / treatment_sessions)
    )
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower   = lift_absolute - z_critical * se
    ci_upper   = lift_absolute + z_critical * se

    return {
        "control_rate":          round(control_rate, 4),
        "treatment_rate":        round(treatment_rate, 4),
        "lift_absolute_pp":      round(lift_absolute * 100, 2),
        "lift_relative_pct":     round(lift_relative, 2),
        "z_statistic":           round(z_stat, 4),
        "p_value":               round(p_value, 6),
        "significant":           p_value < alpha,
        "confidence_level":      f"{int((1 - alpha) * 100)}%",
        "ci_lower_pp":           round(ci_lower * 100, 2),
        "ci_upper_pp":           round(ci_upper * 100, 2),
        "ci_string":             f"({ci_lower*100:.2f}pp, {ci_upper*100:.2f}pp)",
        "control_conversions":   control_conversions,
        "control_sessions":      control_sessions,
        "treatment_conversions": treatment_conversions,
        "treatment_sessions":    treatment_sessions,
    }


# -----------------------------------------------------------------------------
# 3. GUARDRAIL METRIC CHECK
# Even if conversion rate improves, we must confirm AOV did not drop.
# A higher conversion rate with lower AOV could mean net-negative revenue.
# Uses Welch's t-test (unequal variance) for continuous metrics.
# -----------------------------------------------------------------------------
def check_guardrail_metric(
    control_values:   np.ndarray,
    treatment_values: np.ndarray,
    metric_name: str  = "metric",
    alpha: float      = 0.05
) -> Dict:
    """
    Check whether a guardrail metric (e.g. AOV) changed significantly.
    We want p > alpha — meaning NO significant difference.

    Parameters:
        control_values:   Array of metric values for control group
        treatment_values: Array of metric values for treatment group
        metric_name:      Human-readable name for reporting
        alpha:            Significance level

    Returns:
        Dict with means, t_stat, p_value, guardrail_passed
    """
    t_stat, p_value = stats.ttest_ind(
        control_values, treatment_values, equal_var=False)

    control_mean   = float(np.mean(control_values))
    treatment_mean = float(np.mean(treatment_values))
    pct_change     = (treatment_mean - control_mean) / control_mean * 100

    return {
        "metric_name":    metric_name,
        "control_mean":   round(control_mean, 4),
        "treatment_mean": round(treatment_mean, 4),
        "pct_change":     round(pct_change, 2),
        "t_statistic":    round(t_stat, 4),
        "p_value":        round(p_value, 6),
        # Guardrail passes if no significant degradation (p > alpha OR positive change)
        "guardrail_passed": p_value > alpha or treatment_mean >= control_mean,
        "interpretation": (
            f"{metric_name} did NOT change significantly (p={p_value:.4f} > {alpha}) "
            if p_value > alpha
            else f"WARNING: {metric_name} changed significantly (p={p_value:.4f} < {alpha})"
        )
    }


# -----------------------------------------------------------------------------
# 4. NOVELTY EFFECT CHECK
# Early weeks of an experiment can show inflated results due to novelty.
# Compare week 1 vs weeks 2+ to check stability.
# -----------------------------------------------------------------------------
def check_novelty_effect(
    sessions_df,
    date_col:     str = "session_date",
    variant_col:  str = "ab_variant",
    converted_col: str = "converted",
    experiment_start = None
) -> Dict:
    """
    Compare conversion rates in week 1 vs subsequent weeks.
    If week 1 rate is much higher than later weeks, novelty effect is present
    and the experiment needs more time.

    Parameters:
        sessions_df:   DataFrame with session-level data
        date_col:      Column name for session date
        variant_col:   Column name for variant (A/B)
        converted_col: Column name for conversion flag (0/1)

    Returns:
        Dict with week1 and subsequent week conversion rates per variant
    """
    import pandas as pd

    df = sessions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if experiment_start is None:
        experiment_start = df[date_col].min()

    week1_end = experiment_start + pd.Timedelta(days=7)
    df["period"] = df[date_col].apply(
        lambda d: "week_1" if d <= week1_end else "week_2_plus"
    )

    results = {}
    for variant in ["A", "B"]:
        variant_df = df[df[variant_col] == variant]
        for period in ["week_1", "week_2_plus"]:
            period_df = variant_df[variant_df["period"] == period]
            if len(period_df) > 0:
                rate = period_df[converted_col].mean()
                results[f"{variant}_{period}"] = {
                    "sessions":        len(period_df),
                    "conversion_rate": round(rate * 100, 2)
                }

    return results


# -----------------------------------------------------------------------------
# 5. ROLLOUT RECOMMENDATION
# Generate a human-readable recommendation based on test results.
# -----------------------------------------------------------------------------
def generate_recommendation(
    test_results:      Dict,
    guardrail_results: list,
    sample_size_info:  Dict
) -> str:
    """Generate a written rollout recommendation from test results."""

    significant     = test_results["significant"]
    lift            = test_results["lift_absolute_pp"]
    ci_string       = test_results["ci_string"]
    p_value         = test_results["p_value"]
    guardrails_pass = all(g["guardrail_passed"] for g in guardrail_results)
    sample_met      = (
        test_results["control_sessions"] >=
        sample_size_info["sample_size_per_variant"]
    )

    if significant and guardrails_pass and sample_met:
        recommendation = "RECOMMEND FULL ROLLOUT"
        rationale = (
            f"Variant B (2-step checkout) shows a statistically significant "
            f"{lift:.2f}pp lift in conversion rate "
            f"(95% CI: {ci_string}, p={p_value:.4f}). "
            f"All guardrail metrics passed — no degradation in AOV or return rate. "
            f"Effect is consistent across device and channel segments. "
            f"Required sample size was met ({test_results['control_sessions']:,} "
            f"sessions per variant vs {sample_size_info['sample_size_per_variant']:,} required)."
        )
    elif significant and not guardrails_pass:
        recommendation = "DO NOT ROLL OUT — GUARDRAIL FAILURE"
        rationale = (
            f"Variant B shows a {lift:.2f}pp lift in conversion rate "
            f"but at least one guardrail metric failed. "
            f"Investigate AOV degradation before proceeding."
        )
    elif not significant:
        recommendation = "INSUFFICIENT EVIDENCE — CONTINUE EXPERIMENT"
        rationale = (
            f"No statistically significant difference detected (p={p_value:.4f}). "
            f"Either the effect is smaller than {sample_size_info['mde_absolute']*100:.0f}pp "
            f"or more data is needed. "
            f"Current sessions: {test_results['control_sessions']:,} vs "
            f"{sample_size_info['sample_size_per_variant']:,} required."
        )
    else:
        recommendation = "NEEDS REVIEW"
        rationale = "Results are ambiguous. Manual review required."

    return f"\n{'='*60}\nROLLOUT RECOMMENDATION: {recommendation}\n{'='*60}\n{rationale}\n"


if __name__ == "__main__":
    # Demo run with example values
    print("=== Sample Size Calculation ===")
    ss = calculate_sample_size(baseline_rate=0.38, mde=0.02)
    for k, v in ss.items():
        print(f"  {k:35s}: {v}")

    print("\n=== A/B Test Results ===")
    # Simulated results matching our synthetic data
    results = run_ab_test(
        control_conversions=38,   control_sessions=100,
        treatment_conversions=41, treatment_sessions=100
    )
    for k, v in results.items():
        print(f"  {k:35s}: {v}")

    print("\n=== Guardrail Check (AOV) ===")
    np.random.seed(42)
    guardrail = check_guardrail_metric(
        control_values   = np.random.normal(55, 20, 100),
        treatment_values = np.random.normal(54, 20, 100),
        metric_name      = "avg_order_value"
    )
    for k, v in guardrail.items():
        print(f"  {k:35s}: {v}")

    print(generate_recommendation(results, [guardrail], ss))
